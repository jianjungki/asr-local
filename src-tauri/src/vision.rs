use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use image::{imageops::FilterType, DynamicImage, ImageFormat, RgbaImage};
use serde::Serialize;
use std::io::Cursor;

const MAX_CAPTURE_EDGE: u32 = 1800;

#[derive(Clone, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct VisualWindowTarget {
    pub id: String,
    pub title: String,
    pub process_id: u32,
    pub process_name: String,
    pub minimized: bool,
}

#[derive(Clone, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CapturedWindow {
    pub window_id: String,
    pub window_title: String,
    pub process_name: String,
    pub width: u32,
    pub height: u32,
    pub image_data_url: String,
}

#[cfg(windows)]
mod windows_capture {
    use super::*;
    use std::ffi::c_void;
    use std::path::Path;
    use windows::core::PWSTR;
    use windows::Win32::Foundation::{CloseHandle, BOOL, HWND, LPARAM, RECT, TRUE};
    use windows::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDIBits,
        GetWindowDC, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, CAPTUREBLT,
        DIB_RGB_COLORS, HBITMAP, HDC, HGDIOBJ, ROP_CODE, SRCCOPY,
    };
    use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetForegroundWindow, GetWindowRect, GetWindowTextW, GetWindowThreadProcessId,
        IsIconic, IsWindow, IsWindowVisible,
    };

    struct WindowDc {
        hwnd: HWND,
        hdc: HDC,
    }

    impl Drop for WindowDc {
        fn drop(&mut self) {
            unsafe {
                ReleaseDC(self.hwnd, self.hdc);
            }
        }
    }

    struct MemoryDc(HDC);

    impl Drop for MemoryDc {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteDC(self.0);
            }
        }
    }

    struct Bitmap(HBITMAP);

    impl Drop for Bitmap {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteObject(HGDIOBJ(self.0 .0));
            }
        }
    }

    struct SelectedObject {
        hdc: HDC,
        previous: HGDIOBJ,
    }

    impl Drop for SelectedObject {
        fn drop(&mut self) {
            unsafe {
                SelectObject(self.hdc, self.previous);
            }
        }
    }

    struct EnumContext {
        current_process_id: u32,
        windows: Vec<VisualWindowTarget>,
    }

    fn window_text(hwnd: HWND) -> String {
        let mut buffer = vec![0u16; 1024];
        let length = unsafe { GetWindowTextW(hwnd, &mut buffer) };
        if length <= 0 {
            return String::new();
        }
        String::from_utf16_lossy(&buffer[..length as usize])
            .trim()
            .to_string()
    }

    fn window_process_id(hwnd: HWND) -> u32 {
        let mut process_id = 0u32;
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process_id));
        }
        process_id
    }

    fn process_name(process_id: u32) -> String {
        let process =
            match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, process_id) } {
                Ok(process) => process,
                Err(_) => return format!("PID {process_id}"),
            };

        let mut buffer = vec![0u16; 2048];
        let mut length = buffer.len() as u32;
        let queried = unsafe {
            QueryFullProcessImageNameW(
                process,
                PROCESS_NAME_WIN32,
                PWSTR(buffer.as_mut_ptr()),
                &mut length,
            )
        };
        unsafe {
            let _ = CloseHandle(process);
        }

        if queried.is_err() || length == 0 {
            return format!("PID {process_id}");
        }

        let full_path = String::from_utf16_lossy(&buffer[..length as usize]);
        Path::new(&full_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&full_path)
            .trim_end_matches(".exe")
            .to_string()
    }

    fn valid_window_rect(hwnd: HWND) -> Option<RECT> {
        let mut rect = RECT::default();
        if unsafe { GetWindowRect(hwnd, &mut rect) }.is_err() {
            return None;
        }
        let width = rect.right - rect.left;
        let height = rect.bottom - rect.top;
        (width >= 120 && height >= 80).then_some(rect)
    }

    unsafe extern "system" fn enum_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let context = &mut *(lparam.0 as *mut EnumContext);
        if !IsWindowVisible(hwnd).as_bool() || valid_window_rect(hwnd).is_none() {
            return TRUE;
        }

        let title = window_text(hwnd);
        if title.is_empty() {
            return TRUE;
        }

        let process_id = window_process_id(hwnd);
        if process_id == 0 || process_id == context.current_process_id {
            return TRUE;
        }

        context.windows.push(VisualWindowTarget {
            id: (hwnd.0 as usize).to_string(),
            title,
            process_id,
            process_name: process_name(process_id),
            minimized: IsIconic(hwnd).as_bool(),
        });
        TRUE
    }

    pub fn list_windows() -> Result<Vec<VisualWindowTarget>, String> {
        let mut context = EnumContext {
            current_process_id: std::process::id(),
            windows: Vec::new(),
        };

        unsafe {
            EnumWindows(
                Some(enum_window),
                LPARAM((&mut context as *mut EnumContext) as isize),
            )
        }
        .map_err(|error| format!("无法枚举 Windows 窗口：{error}"))?;

        Ok(context.windows)
    }

    fn parse_window_id(window_id: &str) -> Result<HWND, String> {
        let raw = window_id
            .parse::<usize>()
            .map_err(|_| "窗口标识无效，请刷新窗口列表后重试".to_string())?;
        let hwnd = HWND(raw as *mut c_void);
        if raw == 0 || !unsafe { IsWindow(hwnd) }.as_bool() {
            return Err("目标窗口已关闭，请刷新窗口列表后重试".to_string());
        }
        Ok(hwnd)
    }

    fn foreground_or_top_window() -> Result<HWND, String> {
        let foreground = unsafe { GetForegroundWindow() };
        let current_process_id = std::process::id();
        if !foreground.0.is_null()
            && window_process_id(foreground) != current_process_id
            && unsafe { IsWindowVisible(foreground) }.as_bool()
        {
            return Ok(foreground);
        }

        let first = list_windows()?.into_iter().next().ok_or_else(|| {
            "没有找到可捕获的前台窗口，请先打开目标页面或在窗口列表中选择".to_string()
        })?;
        parse_window_id(&first.id)
    }

    fn capture_pixels(hwnd: HWND) -> Result<(u32, u32, Vec<u8>), String> {
        let rect = valid_window_rect(hwnd).ok_or_else(|| "目标窗口尺寸无效".to_string())?;
        let width = rect.right - rect.left;
        let height = rect.bottom - rect.top;
        if width > 12_000 || height > 12_000 {
            return Err("目标窗口尺寸过大，无法安全捕获".to_string());
        }
        if i64::from(width) * i64::from(height) > 40_000_000 {
            return Err("目标窗口像素过多，请缩小窗口后重试".to_string());
        }

        let source_hdc = unsafe { GetWindowDC(hwnd) };
        if source_hdc.0.is_null() {
            return Err("无法读取目标窗口画面".to_string());
        }
        let _source_dc = WindowDc {
            hwnd,
            hdc: source_hdc,
        };

        let memory_hdc = unsafe { CreateCompatibleDC(source_hdc) };
        if memory_hdc.0.is_null() {
            return Err("无法创建截图缓冲区".to_string());
        }
        let _memory_dc = MemoryDc(memory_hdc);

        let bitmap_handle = unsafe { CreateCompatibleBitmap(source_hdc, width, height) };
        if bitmap_handle.0.is_null() {
            return Err("无法创建截图位图".to_string());
        }
        let _bitmap = Bitmap(bitmap_handle);

        let previous = unsafe { SelectObject(memory_hdc, HGDIOBJ(bitmap_handle.0)) };
        if previous.0.is_null() {
            return Err("无法绑定截图位图".to_string());
        }
        let _selected = SelectedObject {
            hdc: memory_hdc,
            previous,
        };

        let printed = unsafe { PrintWindow(hwnd, memory_hdc, PRINT_WINDOW_FLAGS(2)) }.as_bool();
        if !printed {
            unsafe {
                BitBlt(
                    memory_hdc,
                    0,
                    0,
                    width,
                    height,
                    source_hdc,
                    0,
                    0,
                    ROP_CODE(SRCCOPY.0 | CAPTUREBLT.0),
                )
            }
            .map_err(|error| format!("窗口截图失败：{error}"))?;
        }

        let mut bitmap_info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let byte_count = width as usize * height as usize * 4;
        let mut bgra = vec![0u8; byte_count];
        let rows = unsafe {
            GetDIBits(
                memory_hdc,
                bitmap_handle,
                0,
                height as u32,
                Some(bgra.as_mut_ptr() as *mut c_void),
                &mut bitmap_info,
                DIB_RGB_COLORS,
            )
        };
        if rows == 0 {
            return Err("无法读取截图像素".to_string());
        }

        for pixel in bgra.chunks_exact_mut(4) {
            pixel.swap(0, 2);
            pixel[3] = 255;
        }

        Ok((width as u32, height as u32, bgra))
    }

    pub fn capture(window_id: Option<&str>) -> Result<CapturedWindow, String> {
        let hwnd = match window_id.map(str::trim).filter(|value| !value.is_empty()) {
            Some(window_id) => parse_window_id(window_id)?,
            None => foreground_or_top_window()?,
        };
        let window_title = window_text(hwnd);
        let process_id = window_process_id(hwnd);
        let process_name = process_name(process_id);
        let (original_width, original_height, pixels) = capture_pixels(hwnd)?;

        let image = RgbaImage::from_raw(original_width, original_height, pixels)
            .ok_or_else(|| "无法创建截图图像".to_string())?;
        let dynamic = DynamicImage::ImageRgba8(image);
        let resized = if original_width.max(original_height) > MAX_CAPTURE_EDGE {
            dynamic.resize(MAX_CAPTURE_EDGE, MAX_CAPTURE_EDGE, FilterType::Lanczos3)
        } else {
            dynamic
        };
        let width = resized.width();
        let height = resized.height();
        let mut png = Cursor::new(Vec::new());
        resized
            .write_to(&mut png, ImageFormat::Png)
            .map_err(|error| format!("截图编码失败：{error}"))?;
        let encoded = BASE64_STANDARD.encode(png.into_inner());

        Ok(CapturedWindow {
            window_id: (hwnd.0 as usize).to_string(),
            window_title,
            process_name,
            width,
            height,
            image_data_url: format!("data:image/png;base64,{encoded}"),
        })
    }
}

pub fn list_windows() -> Result<Vec<VisualWindowTarget>, String> {
    #[cfg(windows)]
    {
        windows_capture::list_windows()
    }

    #[cfg(not(windows))]
    {
        Err("窗口视觉理解目前仅支持 Windows".to_string())
    }
}

pub fn capture_window(window_id: Option<&str>) -> Result<CapturedWindow, String> {
    #[cfg(windows)]
    {
        windows_capture::capture(window_id)
    }

    #[cfg(not(windows))]
    {
        let _ = window_id;
        Err("窗口视觉理解目前仅支持 Windows".to_string())
    }
}
