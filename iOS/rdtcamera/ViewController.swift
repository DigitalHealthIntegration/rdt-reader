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

    //Camera Capture requiered properties
    var videoDataOutput: AVCaptureVideoDataOutput!
    var videoDataOutputQueue: DispatchQueue!
    var previewLayer:AVCaptureVideoPreviewLayer!
    var captureDevice : AVCaptureDevice!
    let session = AVCaptureSession()
    var objdet=ObjectDetection.init(W: Double(UIScreen.main.bounds.size.width),
                                           H: Double(UIScreen.main.bounds.size.height));
    
    var timeSinceLastCheck = NSDate().timeIntervalSince1970
    var rdtRes = acceptanceStatus(mScale: -1.0, mBrightness: -1, RDT_found: false);
    var goodImageFlag = false;
    
    var wf = UIScreen.main.bounds.size.width/414.0;
    var hf = UIScreen.main.bounds.size.height/896.0;

    
    var resultLabel = UILabel();
    var prompts = UILabel();
    var rdtBox = UIImageView();
    var start_time = NSDate().timeIntervalSince1970
    var returnImg = UIImageView();
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        print(UIScreen.main.bounds.size.width,UIScreen.main.bounds.size.height)

        
        print("\(OpenCVWrapper.openCVVersionString())")
        previewView = UIView(frame: CGRect(x: 0,
                                           y: 0,
                                           width: UIScreen.main.bounds.size.width,
                                           height: UIScreen.main.bounds.size.height))
        
        
        
        resultLabel = UILabel(frame: CGRect(x: 0, y: hf*580, width: wf*100, height: hf*21))
        prompts = UILabel(frame: CGRect(x: 0, y: hf*80, width: 1000, height: hf*100))
        rdtBox = UIImageView(frame: CGRect(x: wf*120, y: hf*120, width: wf*180, height: hf*650))
        returnImg = UIImageView(frame: CGRect(x: 0, y: hf*600, width: wf*40, height: hf*200));

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
        
        let overlay = createOverlay(frame: view.frame,
                                    xOffset: view.frame.midX,
                                    yOffset: view.frame.midY,
                                    rect: CGRect(x: wf*120, y: hf*120, width: wf*180, height: hf*650))
        view.addSubview(overlay)
        //resultLabel.center = CGPoint(x: 160, y: 285)
        //resultLabel.textAlignment = .center
        resultLabel.textColor = UIColor.white
        prompts.textColor = UIColor.white
        prompts.numberOfLines = 4;
        view.addSubview(resultLabel)
        view.addSubview(prompts)
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
    
    
//    @objc func pressed(sender: UIButton!) {
//        print("button clicked");
//        // call rdt api from here
//        hitServer();
//    }
    
    func createOverlay(frame: CGRect,
                       xOffset: CGFloat,
                       yOffset: CGFloat,
                       rect: CGRect) -> UIView {
        // Step 1
        let overlayView = UIView(frame: frame)
        overlayView.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        // Step 2
        let path = CGMutablePath()
//        path.addArc(center: CGPoint(x: xOffset, y: yOffset),
//                    radius: radius,
//                    startAngle: 0.0,
//                    endAngle: 2.0 * .pi,
//                    clockwise: false)
        path.addRect(CGRect(origin: rect.origin, size: rect.size))

        path.addRect(CGRect(origin: .zero, size: overlayView.frame.size))
        // Step 3
        let maskLayer = CAShapeLayer()
        maskLayer.backgroundColor = UIColor.black.cgColor
        maskLayer.path = path
        maskLayer.fillRule = CAShapeLayerFillRule.evenOdd
        // Step 4
        overlayView.layer.mask = maskLayer
        overlayView.clipsToBounds = true

        return overlayView
    }
}


// AVCaptureVideoDataOutputSampleBufferDelegate protocol and related methods
extension ViewController:  AVCaptureVideoDataOutputSampleBufferDelegate{
    
    func hitServer(){
        
        
               CameraFrameHandle.evaluateRDTRestApi(){(result:String,img:String) in
                   print(result);
                   let data = NSData (base64Encoded: img, options: NSData.Base64DecodingOptions(rawValue: 0))
                   let dataStr = Data(result.utf8)
                _ = "error"
                   
                   DispatchQueue.main.async {
                     self.returnImg.image = UIImage(data: data! as Data)
                    self.view.bringSubviewToFront(self.returnImg)

                       do {
                           // make sure this JSON is in the format we expect
                           if let json = try JSONSerialization.jsonObject(with: dataStr, options: []) as? [String: Any] {
                               // try to read out a string array
                               if let names = json["msg"] as? String {
                                   self.resultLabel.text = names
                                
                                self.session.startRunning();
                               }
                           }
                       } catch let error as NSError {
                           print("Failed to load: \(error.localizedDescription)")
                       }
                   }
               }
        DispatchQueue.main.async {
            let data = Data("".utf8)
            self.returnImg.image = UIImage(data: data as Data)
            self.session.stopRunning();

//            self.prompts.text = "Waiting for server"
            self.resultLabel.text = "Waiting"

        }
    }
    
    
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
        let image = UIImage(cgImage: cgImage)
//        print("Image Size \(image.size.width)x\(image.size.height)")
//        print(previewLayer.frame.width,previewLayer.frame.height);
        let curr_time = NSDate().timeIntervalSince1970
        let motion = OpenCVWrapper.checkSteadyStatus(image);
        print("OMG it detects motion",motion);
        if (motion==0){
            
            var tmp = self.objdet.update(imageFrame: image, RDT: &rdtRes)
            //        print("Time taken for full process",NSDate().timeIntervalSince1970-curr_time)

            //        tmp.origin.x = tmp.origin.x/720*414
            tmp.origin.x = self.wf*(tmp.origin.x+60)
            tmp.size.width = self.wf*(tmp.size.width-140)
            tmp.origin.y = self.hf*(tmp.origin.y)
            tmp.size.height = self.hf*(tmp.size.height)
            
            //        tmp.size.height = tmp.size.height-160
                    
                    if curr_time > start_time {
                        start_time = curr_time
                        var textTodisp = "RDT not found"
                        
                        DispatchQueue.main.async {
                            let position = CGPoint(x:tmp.origin.y,y:tmp.origin.x);// CGPoint(x:tmp.origin.x,y:tmp.origin.y);
                            _ = CGSize(width: tmp.size.height, height: tmp.size.width    );// CG
                            if self.rdtRes.RDT_found==false{
                                self.rdtBox.layer.borderColor = UIColor.red.cgColor
                                self.goodImageFlag = false;
                                self.timeSinceLastCheck = NSDate().timeIntervalSince1970
                                textTodisp = "RDT not found";
                                _ = CGPoint(x:0,y:0);// CGPoint(x:tmp.origin.x,y:tmp.origin.y);
                                _ = CGSize(width: 0, height: 0);// CG
                            }
                            else if self.rdtRes.RDT_found==true{
                                self.rdtBox.layer.borderColor = UIColor.green.cgColor

                                self.goodImageFlag = self.rdtRes.mScale>0.6 && self.rdtRes.mBrightness>100 && self.rdtRes.mBrightness<200 && position.x>self.wf*100 && position.x<self.wf*280;
                                _ = curr_time-self.timeSinceLastCheck
                                
                                textTodisp = "Found RDT ..."
                                if self.rdtRes.mScale>0.6{
                                    textTodisp  += "\nscale is good "
                                }
                                else if self.rdtRes.mScale<0.6{
                                    self.timeSinceLastCheck = NSDate().timeIntervalSince1970

                                    textTodisp  += "\nslowly bring camera closer "
                                    
                                }
                                if self.rdtRes.mBrightness>100 && self.rdtRes.mBrightness<200{
                                    textTodisp  += "\nbrightness is good.. hold steady"
                                }
                                else if self.rdtRes.mBrightness<100 {
                                    self.timeSinceLastCheck = NSDate().timeIntervalSince1970

                                    textTodisp  += "\nbrightness is low "
                                }
                                else if self.rdtRes.mBrightness>200 {
                                    self.timeSinceLastCheck = NSDate().timeIntervalSince1970

                                    textTodisp  += "\nbrightness is high "
                                }
                                if position.x<self.wf*100{
                                    self.timeSinceLastCheck = NSDate().timeIntervalSince1970

                                    textTodisp  += "\nmove phone to the left "
                                }
                                else if position.x>self.wf*280{
                                    self.timeSinceLastCheck = NSDate().timeIntervalSince1970

                                    textTodisp  += "\nmove phone to the right "
                                }
                            }
                            self.prompts.text = textTodisp
//                            print("Scale : ",self.rdtRes.mScale, "Brightness : ",self.rdtRes.mBrightness);
            //                Size(width: tmp.size.width, height: tmp.size.height);
//                            self.rdtBox.frame.origin = position
//                            self.rdtBox.frame.size = resolution
                            if curr_time - self.timeSinceLastCheck>1.0{
                                if self.goodImageFlag{
                                    self.timeSinceLastCheck = NSDate().timeIntervalSince1970
                                    CameraFrameHandle.handleFrame(imageFrame: image)
                                    self.hitServer();
                                    AudioServicesPlaySystemSound(1520); // Actuate "Pop" feedback (strong boom)
                                    AudioServicesPlaySystemSound(1108); // Actuate "Pop" feedback (strong boom)

                                    print("\n\n******One second up*****\n\n")
                                }
                                
                                
                            }
                        }
                    }
            
        }
     


        }
    
    // clean up AVCapture
    func stopCamera(){
        session.stopRunning()
    }
    
}
