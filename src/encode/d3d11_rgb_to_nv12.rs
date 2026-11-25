//! BGRA to NV12 on the GPU with our own shaders (Windows, D3D11).
//!
//! The D3D11 video processor used before picks the colour of one pixel for
//! each 2x2 block when it makes the half-resolution colour plane (measured
//! on an AMD RX 6600 XT, 2026-11-25: 1-px yellow lines on black came out with
//! the full yellow's chroma, Cb 16, where the mean is Cb 72). Thin coloured
//! text (a terminal) then loses its colour or bleeds it into the background,
//! depending on which column it sits in. Sunshine converts with shaders that
//! filter the colour (convert_yuv420_packed_uv_type0_ps_*.hlsl); this does
//! the same:
//!
//! - luma: one pass at full size;
//! - colour: one pass at half size, each sample the mean of two rows and of
//!   the columns around the left one with weights 1-2-1 (the "left" chroma
//!   siting H.264 and HEVC decoders assume).
//!
//! The matrix is BT.601 or BT.709, limited or full range, as the stream says.
//! Nothing here touches the CPU: the frame stays a texture.

/// The rows that turn R'G'B' (0..1) into Y', Cb, Cr (0..1): `xyz` are the
/// weights of R, G and B, `w` is added.
pub fn yuv_rows(bt709: bool, full_range: bool) -> [[f32; 4]; 3] {
    let (kr, kb) = if bt709 { (0.2126f32, 0.0722f32) } else { (0.299f32, 0.114f32) };
    let kg = 1.0 - kr - kb;
    // 8-bit: limited range is 16..235 for luma and 16..240 for colour.
    let (y_scale, y_add, c_scale) = if full_range { (1.0, 0.0, 1.0) } else { (219.0 / 255.0, 16.0 / 255.0, 224.0 / 255.0) };
    let c_add = 128.0 / 255.0;
    let u = 0.5 / (1.0 - kb);
    let v = 0.5 / (1.0 - kr);
    [
        [kr * y_scale, kg * y_scale, kb * y_scale, y_add],
        [-kr * u * c_scale, -kg * u * c_scale, (1.0 - kb) * u * c_scale, c_add],
        [(1.0 - kr) * v * c_scale, -kg * v * c_scale, -kb * v * c_scale, c_add],
    ]
}

#[cfg(target_os = "windows")]
pub use gpu::RgbToNv12;

#[cfg(target_os = "windows")]
mod gpu {
    use super::yuv_rows;
    use windows::core::{Interface, PCSTR};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{ID3DBlob, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    const SHADERS: &str = r#"
cbuffer convert : register(b0) {
    float4 y_row;
    float4 u_row;
    float4 v_row;
    float4 texel; // 1/width, 1/height of the source
};
Texture2D image : register(t0);
SamplerState linear_clamp : register(s0);

struct vs_out {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

// One triangle that covers the target.
vs_out vs_main(uint id : SV_VertexID) {
    vs_out o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = t;
    return o;
}

float ps_y(vs_out i) : SV_Target {
    float3 rgb = image.Sample(linear_clamp, i.uv).rgb;
    return dot(rgb, y_row.xyz) + y_row.w;
}

// At half size `uv` falls on the corner of four source pixels: a linear
// sample there is their mean, one a pixel to the left the mean of the four
// before. Together: rows 1-1, columns 1-2-1 around the left column.
float2 ps_uv(vs_out i) : SV_Target {
    float3 right = image.Sample(linear_clamp, i.uv).rgb;
    float3 left = image.Sample(linear_clamp, i.uv - float2(texel.x, 0)).rgb;
    float3 rgb = (left + right) * 0.5;
    return float2(dot(rgb, u_row.xyz) + u_row.w, dot(rgb, v_row.xyz) + v_row.w);
}
"#;

    unsafe fn compile(entry: &[u8], target: &[u8]) -> Result<Vec<u8>, String> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        let result = D3DCompile(
            SHADERS.as_ptr() as *const _,
            SHADERS.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(entry.as_ptr()),
            PCSTR(target.as_ptr()),
            0,
            0,
            &mut code,
            Some(&mut errors),
        );
        if let Err(e) = result {
            let text = errors
                .map(|b| String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize())).into_owned())
                .unwrap_or_default();
            return Err(format!("D3DCompile failed: {e} {text}"));
        }
        let blob = code.ok_or("D3DCompile gave no code")?;
        Ok(std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize()).to_vec())
    }

    struct Target {
        texture: usize,
        slice: u32,
        luma: ID3D11RenderTargetView,
        chroma: ID3D11RenderTargetView,
    }

    /// Converts BGRA textures to NV12 textures of the same size.
    pub struct RgbToNv12 {
        device: ID3D11Device,
        vertex: ID3D11VertexShader,
        luma: ID3D11PixelShader,
        chroma: ID3D11PixelShader,
        sampler: ID3D11SamplerState,
        constants: ID3D11Buffer,
        /// The size the constants were written for.
        size: (u32, u32),
        rows: [[f32; 4]; 3],
        sources: Vec<(usize, ID3D11ShaderResourceView)>,
        targets: Vec<Target>,
    }

    impl RgbToNv12 {
        /// # Safety
        /// `device` is a live D3D11 device.
        pub unsafe fn new(device: &ID3D11Device, bt709: bool, full_range: bool) -> Result<Self, String> {
            let err = |what: &str, e: windows::core::Error| format!("{what} failed: {e}");
            let vs_code = compile(b"vs_main\0", b"vs_4_0\0")?;
            let y_code = compile(b"ps_y\0", b"ps_4_0\0")?;
            let uv_code = compile(b"ps_uv\0", b"ps_4_0\0")?;
            let mut vertex = None;
            device.CreateVertexShader(&vs_code, None, Some(&mut vertex)).map_err(|e| err("CreateVertexShader", e))?;
            let mut luma = None;
            device.CreatePixelShader(&y_code, None, Some(&mut luma)).map_err(|e| err("CreatePixelShader (luma)", e))?;
            let mut chroma = None;
            device.CreatePixelShader(&uv_code, None, Some(&mut chroma)).map_err(|e| err("CreatePixelShader (colour)", e))?;
            let sampler_desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                ComparisonFunc: D3D11_COMPARISON_NEVER,
                MaxLOD: f32::MAX,
                ..Default::default()
            };
            let mut sampler = None;
            device.CreateSamplerState(&sampler_desc, Some(&mut sampler)).map_err(|e| err("CreateSamplerState", e))?;
            let buffer_desc = D3D11_BUFFER_DESC {
                ByteWidth: 64,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                ..Default::default()
            };
            let mut constants = None;
            device.CreateBuffer(&buffer_desc, None, Some(&mut constants)).map_err(|e| err("CreateBuffer", e))?;
            Ok(Self {
                device: device.clone(),
                vertex: vertex.ok_or("no vertex shader")?,
                luma: luma.ok_or("no luma shader")?,
                chroma: chroma.ok_or("no colour shader")?,
                sampler: sampler.ok_or("no sampler")?,
                constants: constants.ok_or("no constant buffer")?,
                size: (0, 0),
                rows: yuv_rows(bt709, full_range),
                sources: Vec::new(),
                targets: Vec::new(),
            })
        }

        unsafe fn source(&mut self, texture: &ID3D11Texture2D) -> Result<ID3D11ShaderResourceView, String> {
            let key = texture.as_raw() as usize;
            if let Some((_, view)) = self.sources.iter().find(|(k, _)| *k == key) {
                return Ok(view.clone());
            }
            let desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                ViewDimension: windows::Win32::Graphics::Direct3D::D3D11_SRV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D11_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1 } },
            };
            let mut view = None;
            self.device
                .CreateShaderResourceView(texture, Some(&desc), Some(&mut view))
                .map_err(|e| format!("CreateShaderResourceView failed: {e}"))?;
            let view = view.ok_or("no shader resource view")?;
            // A capture hands out the same few textures; a new one replaces the oldest.
            if self.sources.len() >= 4 {
                self.sources.remove(0);
            }
            self.sources.push((key, view.clone()));
            Ok(view)
        }

        unsafe fn target(&mut self, texture: &ID3D11Texture2D, slice: u32, array: bool) -> Result<(ID3D11RenderTargetView, ID3D11RenderTargetView), String> {
            let key = texture.as_raw() as usize;
            if let Some(t) = self.targets.iter().find(|t| t.texture == key && t.slice == slice) {
                return Ok((t.luma.clone(), t.chroma.clone()));
            }
            // A plane of an NV12 texture is a render target by its format:
            // R8 is the luma plane, R8G8 the colour plane.
            let view = |format| {
                let desc = if array {
                    D3D11_RENDER_TARGET_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
                        Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2DArray: D3D11_TEX2D_ARRAY_RTV { MipSlice: 0, FirstArraySlice: slice, ArraySize: 1 },
                        },
                    }
                } else {
                    D3D11_RENDER_TARGET_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 { Texture2D: D3D11_TEX2D_RTV { MipSlice: 0 } },
                    }
                };
                let mut out = None;
                self.device
                    .CreateRenderTargetView(texture, Some(&desc), Some(&mut out))
                    .map_err(|e| format!("CreateRenderTargetView failed: {e}"))?;
                out.ok_or_else(|| "no render target view".to_string())
            };
            let luma: ID3D11RenderTargetView = view(DXGI_FORMAT_R8_UNORM)?;
            let chroma: ID3D11RenderTargetView = view(DXGI_FORMAT_R8G8_UNORM)?;
            // An encoder's pool is a few textures (or slices of one).
            if self.targets.len() >= 16 {
                self.targets.remove(0);
            }
            self.targets.push(Target { texture: key, slice, luma: luma.clone(), chroma: chroma.clone() });
            Ok((luma, chroma))
        }

        /// Draws `source` (BGRA, not an array) into `target` (NV12, slice
        /// `slice`). An error means nothing usable was drawn: the caller
        /// converts another way.
        ///
        /// # Safety
        /// The textures belong to this converter's device and `context` is
        /// its immediate context, used from this thread only for the call.
        pub unsafe fn convert(
            &mut self,
            context: &ID3D11DeviceContext,
            source: &ID3D11Texture2D,
            target: &ID3D11Texture2D,
            slice: u32,
        ) -> Result<(), String> {
            let mut src = D3D11_TEXTURE2D_DESC::default();
            let mut dst = D3D11_TEXTURE2D_DESC::default();
            source.GetDesc(&mut src);
            target.GetDesc(&mut dst);
            if src.Format != DXGI_FORMAT_B8G8R8A8_UNORM || dst.Format != DXGI_FORMAT_NV12 {
                return Err(format!("formats {} -> {} are not BGRA -> NV12", src.Format.0, dst.Format.0));
            }
            if src.ArraySize != 1 || src.SampleDesc.Count != 1 || src.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
                return Err("the source cannot be sampled (an array, multisampled or without the shader-resource flag)".into());
            }
            if dst.BindFlags & D3D11_BIND_RENDER_TARGET.0 as u32 == 0 {
                return Err("the target is no render target".into());
            }
            if (src.Width, src.Height) != (dst.Width, dst.Height) || src.Width % 2 != 0 || src.Height % 2 != 0 {
                return Err(format!("sizes {}x{} -> {}x{}", src.Width, src.Height, dst.Width, dst.Height));
            }
            let image = self.source(source)?;
            let (luma, chroma) = self.target(target, slice, dst.ArraySize > 1)?;

            if self.size != (src.Width, src.Height) {
                let mut data = [0f32; 16];
                data[..4].copy_from_slice(&self.rows[0]);
                data[4..8].copy_from_slice(&self.rows[1]);
                data[8..12].copy_from_slice(&self.rows[2]);
                data[12] = 1.0 / src.Width as f32;
                data[13] = 1.0 / src.Height as f32;
                context.UpdateSubresource(&self.constants, 0, None, data.as_ptr() as *const _, 0, 0);
                self.size = (src.Width, src.Height);
            }

            // Nothing else draws with this context (the capture copies), so
            // the state is set whole each time and nothing is restored.
            context.IASetInputLayout(None);
            context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context.VSSetShader(&self.vertex, None);
            context.RSSetState(None);
            context.OMSetBlendState(None, None, 0xffff_ffff);
            context.OMSetDepthStencilState(None, 0);
            context.PSSetConstantBuffers(0, Some(&[Some(self.constants.clone())]));
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            context.PSSetShaderResources(0, Some(&[Some(image)]));

            let viewport = |w: u32, h: u32| D3D11_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 };
            context.OMSetRenderTargets(Some(&[Some(luma)]), None);
            context.RSSetViewports(Some(&[viewport(src.Width, src.Height)]));
            context.PSSetShader(&self.luma, None);
            context.Draw(3, 0);

            context.OMSetRenderTargets(Some(&[Some(chroma)]), None);
            context.RSSetViewports(Some(&[viewport(src.Width / 2, src.Height / 2)]));
            context.PSSetShader(&self.chroma, None);
            context.Draw(3, 0);

            // Let go of both textures: the capture writes the one, the
            // encoder reads the other.
            context.OMSetRenderTargets(None, None);
            context.PSSetShaderResources(0, Some(&[None]));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::yuv_rows;

    fn convert(rows: &[[f32; 4]; 3], rgb: [u8; 3]) -> [u8; 3] {
        let c = rgb.map(|v| v as f32 / 255.0);
        [0, 1, 2].map(|i| ((rows[i][0] * c[0] + rows[i][1] * c[1] + rows[i][2] * c[2] + rows[i][3]) * 255.0).round() as u8)
    }

    #[test]
    fn bt709_limited_matches_the_standard() {
        let rows = yuv_rows(true, false);
        assert_eq!(convert(&rows, [255, 0, 0]), [63, 102, 240]);
        assert_eq!(convert(&rows, [0, 255, 0]), [173, 42, 26]);
        assert_eq!(convert(&rows, [0, 0, 255]), [32, 240, 118]);
        assert_eq!(convert(&rows, [128, 128, 128]), [126, 128, 128]);
        assert_eq!(convert(&rows, [0, 0, 0]), [16, 128, 128]);
        assert_eq!(convert(&rows, [255, 255, 255]), [235, 128, 128]);
    }

    #[test]
    fn bt601_and_full_range() {
        assert_eq!(convert(&yuv_rows(false, false), [255, 0, 0]), [81, 90, 240]);
        assert_eq!(convert(&yuv_rows(true, true), [255, 255, 255]), [255, 128, 128]);
        assert_eq!(convert(&yuv_rows(true, true), [0, 0, 0]), [0, 128, 128]);
        assert_eq!(convert(&yuv_rows(false, true), [255, 0, 0]), [76, 85, 255]);
    }
}
