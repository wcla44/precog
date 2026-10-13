//! Native AVFoundation webcam capture (nokhwa's macOS backend never delivers frames on recent macOS).
//! Frames arrive as BGRA on a dispatch queue and are converted to RGB.

use anyhow::{anyhow, bail, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject};
use objc2::{define_class, msg_send, AllocAnyThread, DefinedClass};
use objc2_av_foundation::{
    AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceInput, AVCaptureOutput, AVCaptureSession,
    AVCaptureSessionPreset1280x720, AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate,
    AVMediaTypeVideo,
};
use objc2_core_media::CMSampleBuffer;
use objc2_core_video::{
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use std::time::Duration;

pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

pub struct Ivars {
    tx: Sender<RgbFrame>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[name = "PrecogCameraDelegate"]
    #[ivars = Ivars]
    struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for Delegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn did_output(&self, _output: &AVCaptureOutput, sample: &CMSampleBuffer, _conn: &AVCaptureConnection) {
            let Some(pb) = (unsafe { sample.image_buffer() }) else { return };
            unsafe { CVPixelBufferLockBaseAddress(&pb, CVPixelBufferLockFlags::ReadOnly) };
            let (w, h, bpr) = (CVPixelBufferGetWidth(&pb), CVPixelBufferGetHeight(&pb), CVPixelBufferGetBytesPerRow(&pb));
            let base = CVPixelBufferGetBaseAddress(&pb) as *const u8;
            if !base.is_null() {
                let src = unsafe { std::slice::from_raw_parts(base, bpr * h) };
                let mut data = Vec::with_capacity(w * h * 3);
                for row in src.chunks_exact(bpr) {
                    for px in row[..w * 4].chunks_exact(4) {
                        data.extend_from_slice(&[px[2], px[1], px[0]]); // BGRA -> RGB
                    }
                }
                let _ = self.ivars().tx.try_send(RgbFrame { width: w, height: h, data }); // drop if the consumer lags
            }
            unsafe { CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags::ReadOnly) };
        }
    }
);

impl Delegate {
    fn new(tx: Sender<RgbFrame>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(Ivars { tx });
        unsafe { msg_send![super(this), init] }
    }
}

pub struct Camera {
    session: Retained<AVCaptureSession>,
    _output: Retained<AVCaptureVideoDataOutput>,
    _delegate: Retained<Delegate>,
    _queue: dispatch2::DispatchRetained<dispatch2::DispatchQueue>,
    rx: Receiver<RgbFrame>,
}

// The AVFoundation objects are only touched at setup/teardown; frames come through the channel.
unsafe impl Send for Camera {}

impl Camera {
    pub fn open(index: u32) -> Result<Self> {
        unsafe {
            let media = AVMediaTypeVideo.ok_or_else(|| anyhow!("AVMediaTypeVideo unavailable"))?;
            #[allow(deprecated)]
            let device = if index == 0 {
                AVCaptureDevice::defaultDeviceWithMediaType(media)
            } else {
                let all = AVCaptureDevice::devicesWithMediaType(media);
                ((index as usize) < all.count()).then(|| all.objectAtIndex(index as usize))
            }
            .ok_or_else(|| anyhow!("no camera #{index}"))?;
            let input = AVCaptureDeviceInput::deviceInputWithDevice_error(&device)
                .map_err(|e| anyhow!("cannot open camera: {}", e.localizedDescription()))?;
            let session = AVCaptureSession::new();
            if session.canSetSessionPreset(AVCaptureSessionPreset1280x720) {
                session.setSessionPreset(AVCaptureSessionPreset1280x720);
            }
            if !session.canAddInput(&input) {
                bail!("camera input rejected by the capture session");
            }
            session.addInput(&input);

            let output = AVCaptureVideoDataOutput::new();
            let key: &NSString = &*(kCVPixelBufferPixelFormatTypeKey as *const _ as *const NSString);
            let fmt = NSNumber::new_u32(kCVPixelFormatType_32BGRA);
            let settings = NSDictionary::<NSString, AnyObject>::from_slices(&[key], &[&*fmt as &AnyObject]);
            output.setVideoSettings(Some(&settings));
            output.setAlwaysDiscardsLateVideoFrames(true);
            let (tx, rx) = bounded(2);
            let delegate = Delegate::new(tx);
            let queue = dispatch2::DispatchQueue::new("precog.camera", None);
            output.setSampleBufferDelegate_queue(Some(ProtocolObject::from_ref(&*delegate)), Some(&queue));
            if !session.canAddOutput(&output) {
                bail!("video output rejected by the capture session");
            }
            session.addOutput(&output);
            session.startRunning();
            Ok(Self { session, _output: output, _delegate: delegate, _queue: queue, rx })
        }
    }

    pub fn frame(&self) -> Result<RgbFrame> {
        self.rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| anyhow!("no frame from the camera for 5 s (is another app using it?)"))
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        unsafe { self.session.stopRunning() };
    }
}
