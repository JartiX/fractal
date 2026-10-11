use serde::Serialize;

#[derive(Serialize)]
struct Gpu {
  name: String,
  /// dedicated video memory, bytes
  vram: u64,
}

#[derive(Serialize)]
struct SystemMemory {
  gpus: Vec<Gpu>,
  /// physical memory, bytes (0 if unknown)
  ram: u64,
}

/// What the machine has, so that the page can offer sensible memory limits:
/// a browser by itself is told neither figure.
#[tauri::command]
fn system_memory() -> SystemMemory {
  #[cfg(windows)]
  {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE};
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut gpus = Vec::new();
    let mut ram = 0u64;
    unsafe {
      if let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() {
        let mut i = 0;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
          i += 1;
          let Ok(desc) = adapter.GetDesc1() else { continue };
          if desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
            continue;
          }
          let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
          gpus.push(Gpu {
            name: String::from_utf16_lossy(&desc.Description[..len]),
            vram: desc.DedicatedVideoMemory as u64,
          });
        }
      }
      let mut status = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
      if GlobalMemoryStatusEx(&mut status).is_ok() {
        ram = status.ullTotalPhys;
      }
    }
    SystemMemory { gpus, ram }
  }
  #[cfg(not(windows))]
  {
    SystemMemory { gpus: Vec::new(), ram: 0 }
  }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![system_memory])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while building tauri application");
}
