//! The FX materials' samplers on Metal (macOS).
//!
//! Metal binds at most 16 samplers per shader stage (wgpu `max_samplers_per_shader_stage`), and Bevy's view bind groups
//! already use up to 12 of them. [`FxMaterial`](crate::FxMaterial) binds one sampler per texture slot (19) and
//! [`FxCarMaterial`](crate::car_material::FxCarMaterial) 13, so on Metal their layouts fail wgpu validation and Bevy quits
//! on the first draw (the sky's cloud layers and imported maps still use FxMaterial under Remaster).
//!
//! Where [`ENABLED`] (Apple targets), the per-slot samplers are bound as one `binding_array<sampler, 19>` at [`BINDING`]
//! instead: element tf for 2D slot tf, 16 + k for cube slot k, each still the slot's own image sampler (slots a material
//! type doesn't have repeat another of its samplers). Binding-array samplers count against Metal's argument-buffer
//! limit (1024+ on Apple Silicon), not the 16, and need `TEXTURE_BINDING_ARRAY` (Apple Silicon on macOS 13+). The
//! generated programs and the FX prepass shader index the array ([`wgsl`]). Other platforms keep the derived per-slot
//! layout unchanged.

use std::num::NonZeroU32;

use bevy::log::error;
use bevy::render::render_resource::{
    AsBindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, OwnedBindingResource,
    PreparedBindGroup, SamplerBindingType, SamplerDescriptor, ShaderStages, UnpreparedBindGroup, WgpuFeatures,
};
use bevy::render::renderer::RenderDevice;

use crate::program::{CUBE_SLOTS, TEX_2D_BASE, TEX_CUBE_BASE};

/// True where the FX materials bind their samplers as one array (Apple targets: Metal, also under MoltenVK).
pub const ENABLED: bool = cfg!(target_vendor = "apple");
/// The array's binding (slot tf0's sampler in the per-slot layout).
pub const BINDING: u32 = TEX_2D_BASE + 1;
/// 16 2D slots + the cube slots.
pub const COUNT: u32 = 16 + CUBE_SLOTS as u32;

/// The array element standing for per-slot sampler binding `binding` (`TEX_2D_BASE + 2 tf + 1` or
/// `TEX_CUBE_BASE + 2 k + 1`), if it is one.
pub fn element(binding: u32) -> Option<u32> {
    if binding > TEX_2D_BASE && binding < TEX_2D_BASE + 32 && (binding - TEX_2D_BASE) % 2 == 1 {
        Some((binding - TEX_2D_BASE) / 2)
    } else if binding > TEX_CUBE_BASE && binding < TEX_CUBE_BASE + 2 * CUBE_SLOTS as u32 && (binding - TEX_CUBE_BASE) % 2 == 1 {
        Some(16 + (binding - TEX_CUBE_BASE) / 2)
    } else {
        None
    }
}

/// `entries` (a derived per-slot FX layout) with the slot samplers replaced by the array, where [`ENABLED`].
pub fn layout_entries(entries: Vec<BindGroupLayoutEntry>) -> Vec<BindGroupLayoutEntry> {
    if ENABLED { array_layout(entries) } else { entries }
}

fn array_layout(entries: Vec<BindGroupLayoutEntry>) -> Vec<BindGroupLayoutEntry> {
    let mut visibility = ShaderStages::NONE;
    let mut out: Vec<BindGroupLayoutEntry> = entries
        .into_iter()
        .filter(|e| {
            let slot = matches!(e.ty, BindingType::Sampler(_)) && element(e.binding).is_some();
            if slot {
                visibility |= e.visibility;
            }
            !slot
        })
        .collect();
    if !visibility.is_empty() {
        out.push(BindGroupLayoutEntry {
            binding: BINDING,
            visibility,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: NonZeroU32::new(COUNT),
        });
    }
    out
}

/// The bind group for a derived per-slot `unprepared` group on `layout` (from [`layout_entries`]).
pub fn prepare(label: &'static str, layout: &BindGroupLayout, device: &RenderDevice, unprepared: UnpreparedBindGroup) -> PreparedBindGroup {
    let bindings = unprepared.bindings;
    let bind_group = {
        let mut samplers = vec![None; COUNT as usize];
        let mut entries = Vec::new();
        for (binding, resource) in bindings.iter() {
            match (ENABLED.then(|| element(*binding)).flatten(), resource) {
                (Some(i), OwnedBindingResource::Sampler(_, s)) => samplers[i as usize] = Some(&**s),
                _ => entries.push(BindGroupEntry { binding: *binding, resource: resource.get_binding() }),
            }
        }
        let fallback;
        if ENABLED {
            let first = match samplers.iter().flatten().next() {
                Some(&s) => s,
                None => {
                    fallback = device.create_sampler(&SamplerDescriptor::default());
                    &*fallback
                }
            };
            let array: Vec<_> = samplers.iter().map(|s| s.unwrap_or(first)).collect();
            entries.push(BindGroupEntry { binding: BINDING, resource: BindingResource::SamplerArray(&array) });
            device.create_bind_group(label, layout, &entries)
        } else {
            device.create_bind_group(label, layout, &entries)
        }
    };
    PreparedBindGroup { bindings, bind_group }
}

/// [`AsBindGroup::bind_group_layout_entries`] for an FX material whose resources `B` binds.
pub fn entries<B: AsBindGroup>(device: &RenderDevice, force_no_bindless: bool) -> Vec<BindGroupLayoutEntry> {
    layout_entries(B::bind_group_layout_entries(device, force_no_bindless))
}

/// A WGSL module declaring the per-slot FX samplers (`var fx_t<tf>_s: sampler;`, `var fx_cube<k>_s: sampler;`) rewritten
/// for the array: the first such declaration becomes the `fx_samplers` array, the others go, and every use indexes the
/// array. Unchanged where not [`ENABLED`].
pub fn wgsl(module: &str) -> String {
    if ENABLED { array_wgsl(module) } else { module.to_owned() }
}

fn array_wgsl(module: &str) -> String {
    let mut out = String::with_capacity(module.len());
    let mut declared = false;
    for line in module.split_inclusive('\n') {
        let decl = line.split_once(" var ").and_then(|(_, rest)| rest.trim_end().strip_suffix(": sampler;")).and_then(slot_element);
        if decl.is_some() {
            if !declared {
                let head = &line[..line.find(" var ").unwrap()];
                let head = head.rsplit_once("@binding(").map_or(head, |(h, _)| h);
                out += &format!("{head}@binding({BINDING}) var fx_samplers: binding_array<sampler, {COUNT}>;\n");
                declared = true;
            }
            continue;
        }
        out += &replace_uses(line);
    }
    out
}

/// The array element a per-slot sampler name (`fx_t<tf>_s`, `fx_cube<k>_s`) stands for.
fn slot_element(name: &str) -> Option<u32> {
    let n: u32 = if let Some(tf) = name.strip_prefix("fx_t").and_then(|s| s.strip_suffix("_s")) {
        tf.parse().ok().filter(|&tf| tf < 16)?
    } else {
        16 + name.strip_prefix("fx_cube").and_then(|s| s.strip_suffix("_s"))?.parse::<u32>().ok().filter(|&k| k < CUBE_SLOTS as u32)?
    };
    Some(n)
}

/// `line` with each per-slot sampler identifier replaced by its array element.
fn replace_uses(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find(|c: char| c.is_ascii_alphabetic() || c == '_') {
        out += &rest[..start];
        let len = rest[start..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len() - start);
        let ident = &rest[start..start + len];
        match slot_element(ident) {
            Some(i) => out += &format!("fx_samplers[{i}]"),
            None => out += ident,
        }
        rest = &rest[start + len..];
    }
    out + rest
}

/// Logs why the game is about to stop when this GPU can't bind the array (Intel Macs, macOS 12 and older).
pub fn check(device: &RenderDevice) {
    if ENABLED && !device.features().contains(WgpuFeatures::TEXTURE_BINDING_ARRAY) {
        error!(
            "this GPU has no Metal sampler arrays (wgpu TEXTURE_BINDING_ARRAY): FH1 Rewrite needs an Apple Silicon Mac on macOS 14 \
             or newer; the game's materials will fail to bind"
        );
    }
}

/// `AsBindGroup` for an FX material `$material` whose bound resources are its derived `bindings: $bindings` field: the
/// derived layout and bind group, with the slot samplers turned into the array where [`ENABLED`].
macro_rules! fx_as_bind_group {
    ($material:ident, $bindings:ty, $data:ty) => {
        impl bevy::render::render_resource::AsBindGroup for $material {
            type Data = $data;
            type Param = <$bindings as bevy::render::render_resource::AsBindGroup>::Param;

            fn label() -> &'static str {
                stringify!($material)
            }

            fn bind_group_data(&self) -> Self::Data {
                <$data>::from(self)
            }

            fn unprepared_bind_group(
                &self,
                layout: &bevy::render::render_resource::BindGroupLayout,
                render_device: &bevy::render::renderer::RenderDevice,
                param: &mut bevy::ecs::system::SystemParamItem<'_, '_, Self::Param>,
                force_no_bindless: bool,
            ) -> Result<bevy::render::render_resource::UnpreparedBindGroup, bevy::render::render_resource::AsBindGroupError> {
                if $crate::sampler_array::ENABLED {
                    // A sampler array has no `OwnedBindingResource`: Bevy calls `as_bind_group` instead.
                    return Err(bevy::render::render_resource::AsBindGroupError::CreateBindGroupDirectly);
                }
                self.bindings.unprepared_bind_group(layout, render_device, param, force_no_bindless)
            }

            fn as_bind_group(
                &self,
                layout_descriptor: &bevy::render::render_resource::BindGroupLayoutDescriptor,
                render_device: &bevy::render::renderer::RenderDevice,
                pipeline_cache: &bevy::render::render_resource::PipelineCache,
                param: &mut bevy::ecs::system::SystemParamItem<'_, '_, Self::Param>,
            ) -> Result<bevy::render::render_resource::PreparedBindGroup, bevy::render::render_resource::AsBindGroupError> {
                let layout = pipeline_cache.get_bind_group_layout(layout_descriptor);
                let unprepared = self.bindings.unprepared_bind_group(&layout, render_device, param, false)?;
                Ok($crate::sampler_array::prepare(Self::label(), &layout, render_device, unprepared))
            }

            fn bind_group_layout_entries(
                render_device: &bevy::render::renderer::RenderDevice,
                force_no_bindless: bool,
            ) -> Vec<bevy::render::render_resource::BindGroupLayoutEntry> {
                $crate::sampler_array::entries::<$bindings>(render_device, force_no_bindless)
            }
        }
    };
}
pub(crate) use fx_as_bind_group;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elements() {
        assert_eq!(element(3), Some(0));
        assert_eq!(element(33), Some(15));
        assert_eq!(element(35), Some(16));
        assert_eq!(element(39), Some(18));
        for b in [0, 1, 2, 4, 32, 34, 40, 41] {
            assert_eq!(element(b), None, "binding {b}");
        }
        assert_eq!(slot_element("fx_t0_s"), Some(0));
        assert_eq!(slot_element("fx_t15_s"), Some(15));
        assert_eq!(slot_element("fx_cube2_s"), Some(18));
        assert_eq!(slot_element("fx_t16_s"), None);
        assert_eq!(slot_element("fx_t1"), None);
    }

    #[test]
    fn rewrites_wgsl() {
        let src = "@group(#{MATERIAL_BIND_GROUP}) @binding(2) var fx_t0: texture_2d<f32>;\n\
                   @group(#{MATERIAL_BIND_GROUP}) @binding(3) var fx_t0_s: sampler;\n\
                   @group(#{MATERIAL_BIND_GROUP}) @binding(23) var fx_t10_s: sampler;\n\
                   @group(#{MATERIAL_BIND_GROUP}) @binding(35) var fx_cube0_s: sampler;\n\
                   let a = textureSample(fx_t10, fx_t10_s, uv) + textureSample(fx_t1, fx_t1_s, uv);\n\
                   let b = textureSample(fx_cube0, fx_cube0_s, d); // fx_t0_sx stays\n";
        assert_eq!(wgsl(src), if ENABLED { array_wgsl(src) } else { src.to_owned() });
        assert_eq!(
            array_wgsl(src),
            "@group(#{MATERIAL_BIND_GROUP}) @binding(2) var fx_t0: texture_2d<f32>;\n\
             @group(#{MATERIAL_BIND_GROUP}) @binding(3) var fx_samplers: binding_array<sampler, 19>;\n\
             let a = textureSample(fx_t10, fx_samplers[10], uv) + textureSample(fx_t1, fx_samplers[1], uv);\n\
             let b = textureSample(fx_cube0, fx_samplers[16], d); // fx_t0_sx stays\n"
        );
    }

    #[test]
    fn array_layout_replaces_slot_samplers() {
        let entry = |binding, ty, visibility| BindGroupLayoutEntry { binding, visibility, ty, count: None };
        let tex = BindingType::Texture {
            sample_type: bevy::render::render_resource::TextureSampleType::Float { filterable: true },
            view_dimension: bevy::render::render_resource::TextureViewDimension::D2,
            multisampled: false,
        };
        let samp = BindingType::Sampler(SamplerBindingType::Filtering);
        let out = array_layout(vec![
            entry(2, tex, ShaderStages::FRAGMENT),
            entry(3, samp, ShaderStages::FRAGMENT),
            entry(4, tex, ShaderStages::FRAGMENT),
            entry(5, samp, ShaderStages::VERTEX),
            entry(39, samp, ShaderStages::FRAGMENT),
        ]);
        assert_eq!(out.iter().map(|e| e.binding).collect::<Vec<_>>(), [2, 4, BINDING]);
        let array = out.last().unwrap();
        assert_eq!(array.count, NonZeroU32::new(COUNT));
        assert_eq!(array.visibility, ShaderStages::VERTEX_FRAGMENT);
    }
}
