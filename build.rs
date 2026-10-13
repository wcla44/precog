fn main() {
    // onnxruntime's prebuilt CoreML provider references classes added in macOS 14.4. With an older Xcode SDK
    // they are missing at link time: let the dynamic loader resolve them (they exist at runtime on 14.4+).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        for sym in ["_OBJC_CLASS_$_MLComputePlan", "_OBJC_CLASS_$_MLOptimizationHints"] {
            println!("cargo:rustc-link-arg=-Wl,-U,{sym}");
        }
    }
}
