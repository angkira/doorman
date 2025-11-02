use anyhow::Result;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera as NokhwaCamera,
};

fn main() -> Result<()> {
    println!("🎥 Testing camera access...\n");
    
    // Try to open camera 0
    let index = CameraIndex::Index(0);
    let requested = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::AbsoluteHighestResolution
    );
    
    println!("Opening camera at index 0...");
    let mut camera = NokhwaCamera::new(index, requested)?;
    
    println!("✅ Camera opened successfully!");
    println!("   Resolution: {:?}", camera.resolution());
    println!("   Frame format: {:?}", camera.frame_format());
    
    // Start the camera stream
    println!("\nStarting camera stream...");
    camera.open_stream()?;
    println!("✅ Stream started!");
    
    // Try to capture a frame
    println!("\nCapturing test frame...");
    let frame = camera.frame()?;
    println!("✅ Frame captured successfully!");
    println!("   Frame size: {} bytes", frame.buffer().len());
    
    // Decode the frame
    println!("\nDecoding frame to RGB...");
    let decoded = frame.decode_image::<RgbFormat>()?;
    let (width, height) = (decoded.width(), decoded.height());
    println!("✅ Frame decoded successfully!");
    println!("   Image size: {}x{}", width, height);
    
    println!("\n🎉 Camera test passed! Your camera is working correctly.");
    
    Ok(())
}

