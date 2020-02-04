//
//  ViewController.swift
//  RDT Camera
//
//  Created by Akhil Kumar on 24/10/19.
//  Copyright © 2019 IPRD. All rights reserved.
//

import UIKit
import AVFoundation
import AudioToolbox

class ViewController: UIViewController{
    var previewView : UIView!
    var boxView:UIView!
    var actionButton = UIButton()
    var resultLabel = UILabel(frame: CGRect(x: 0, y: 580, width: 200, height: 21))
    //Camera Capture requiered properties
    var videoDataOutput: AVCaptureVideoDataOutput!
    var videoDataOutputQueue: DispatchQueue!
    var previewLayer:AVCaptureVideoPreviewLayer!
    var captureDevice : AVCaptureDevice!
    let session = AVCaptureSession()
    var rdtBox = UIImageView(frame: CGRect(x: 0, y: 0, width: 0, height: 0))
    var start_time = NSDate().timeIntervalSince1970
    var returnImg = UIImageView(frame: CGRect(x: 0, y: 600, width: 40, height: 200));
    
    
   
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        print("\(OpenCVWrapper.openCVVersionString())")
        previewView = UIView(frame: CGRect(x: 0,
                                           y: 0,
                                           width: UIScreen.main.bounds.size.width,
                                           height: UIScreen.main.bounds.size.height))
        previewView.contentMode = UIView.ContentMode.scaleAspectFit
        returnImg.contentMode = UIImageView.ContentMode.scaleAspectFit
        previewView.backgroundColor = UIColor.darkGray
        view.addSubview(previewView)
        
        //Add a view on top of the cameras' view
        boxView = UIView(frame: self.previewView.frame)
        view.addSubview(boxView)
        rdtBox.backgroundColor=UIColor.clear
        rdtBox.layer.borderWidth=5.0
        rdtBox.layer.borderColor=UIColor.red.cgColor
        rdtBox.contentMode = UIView.ContentMode.scaleAspectFit
        returnImg.backgroundColor = UIColor.black
        view.addSubview(rdtBox);
        
        view.addSubview(returnImg);
        actionButton.setTitle("Check RDT", for: .normal);
        actionButton.setTitleColor(UIColor.blue, for: .normal)
        actionButton.frame = CGRect(x: 10, y: UIScreen.main.bounds.size.height - 80, width: UIScreen.main.bounds.size.width - 20, height: 60)
        actionButton.addTarget(self, action: #selector(pressed(sender:)), for: .touchUpInside)
        actionButton.backgroundColor = UIColor.init(red: 0.5, green: 0.5, blue: 0.5, alpha: 0.5)
        
        view.addSubview(actionButton);
        //resultLabel.center = CGPoint(x: 160, y: 285)
        //resultLabel.textAlignment = .center
        view.addSubview(resultLabel)

        print(previewView.layer.bounds)
        print(view.layer.bounds)
        
        self.setupAVCapture()
    }
    
    override var shouldAutorotate: Bool {
        if (UIDevice.current.orientation == UIDeviceOrientation.landscapeLeft ||
            UIDevice.current.orientation == UIDeviceOrientation.landscapeRight ||
            UIDevice.current.orientation == UIDeviceOrientation.unknown) {
            return false
        }
        else {
            return true
        }
    }
    
    
    @objc func pressed(sender: UIButton!) {
        print("button clicked");
        // call rdt api from here
        AudioServicesPlaySystemSound(1520); // Actuate "Pop" feedback (strong boom)
        DispatchQueue.main.async {
            let data = Data("".utf8)
            self.returnImg.image = UIImage(data: data as Data)
            self.resultLabel.text = "waiting"
        }
        CameraFrameHandle.evaluateRDTRestApi(){(result:String,img:String) in
            print(result);
            let data = NSData (base64Encoded: img, options: NSData.Base64DecodingOptions(rawValue: 0))
            let dataStr = Data(result.utf8)
            var res = "error"
            
            DispatchQueue.main.async {
              self.returnImg.image = UIImage(data: data! as Data)
                do {
                    // make sure this JSON is in the format we expect
                    if let json = try JSONSerialization.jsonObject(with: dataStr, options: []) as? [String: Any] {
                        // try to read out a string array
                        if let names = json["msg"] as? String {
                            self.resultLabel.text = names

                        }
                    }
                } catch let error as NSError {
                    print("Failed to load: \(error.localizedDescription)")
                }
            }
        }
    }
}


// AVCaptureVideoDataOutputSampleBufferDelegate protocol and related methods
extension ViewController:  AVCaptureVideoDataOutputSampleBufferDelegate{
    
    func setupAVCapture(){
        session.sessionPreset = AVCaptureSession.Preset.hd1280x720
        guard let device = AVCaptureDevice
            .default(AVCaptureDevice.DeviceType.builtInWideAngleCamera,
                     for: .video,
                     position: AVCaptureDevice.Position.back) else {
                        return
        }
        captureDevice = device
        beginSession()
    }
    
    func beginSession(){
        var deviceInput: AVCaptureDeviceInput!
        
        do {
            deviceInput = try AVCaptureDeviceInput(device: captureDevice)
            guard deviceInput != nil else {
                print("error: cant get deviceInput")
                return
            }
            
            if self.session.canAddInput(deviceInput){
                self.session.addInput(deviceInput)
            }
            
            videoDataOutput = AVCaptureVideoDataOutput()
            videoDataOutput.alwaysDiscardsLateVideoFrames=true
            videoDataOutputQueue = DispatchQueue(label: "VideoDataOutputQueue")
            videoDataOutput.setSampleBufferDelegate(self, queue:self.videoDataOutputQueue)
            
            if session.canAddOutput(self.videoDataOutput){
                session.addOutput(self.videoDataOutput)
            }
            
            videoDataOutput.connection(with: .video)?.isEnabled = true
            
            previewLayer = AVCaptureVideoPreviewLayer(session: self.session)
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect
            let rootLayer :CALayer = self.previewView.layer
            rootLayer.masksToBounds=true
            previewLayer.frame = rootLayer.bounds
            rootLayer.addSublayer(self.previewLayer)
            session.startRunning()
        } catch let error as NSError {
            deviceInput = nil
            print("error: \(error.localizedDescription)")
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // do stuff here
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return  }
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return  }
        var rdtRes: [Bool] = [false]
        let image = UIImage(cgImage: cgImage)
        CameraFrameHandle.handleFrame(imageFrame: image)
//        print("Image Size \(image.size.width)x\(image.size.height)")
//        print(previewLayer.frame.width,previewLayer.frame.height);
        let curr_time = NSDate().timeIntervalSince1970
        
        var tmp = ObjectDetection.update(imageFrame: image, RDT: &rdtRes)
//        print("Time taken for full process",NSDate().timeIntervalSince1970-curr_time)

//        tmp.origin.x = tmp.origin.x/720*414
        tmp.origin.x = tmp.origin.x+70
        tmp.size.width = tmp.size.width-140
//        tmp.size.height = tmp.size.height-160
        
        if curr_time > start_time {
            start_time = curr_time
            DispatchQueue.main.async {
                var position = CGPoint(x:tmp.origin.y,y:tmp.origin.x);// CGPoint(x:tmp.origin.x,y:tmp.origin.y);
                var resolution = CGSize(width: tmp.size.height, height: tmp.size.width	);// CG
                if rdtRes[0]==false{
                     position = CGPoint(x:0,y:0);// CGPoint(x:tmp.origin.x,y:tmp.origin.y);
                     resolution = CGSize(width: 0, height: 0);// CG
                }
//                Size(width: tmp.size.width, height: tmp.size.height);
                self.rdtBox.frame.origin = position
                self.rdtBox.frame.size = resolution
            }
        }


	    }
    
    // clean up AVCapture
    func stopCamera(){
        session.stopRunning()
    }
    
}
