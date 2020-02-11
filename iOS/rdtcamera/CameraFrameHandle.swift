//
//  CameraFrameHandle.swift
//  RDT Camera
//
//  Created by developer on 24/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

import UIKit
import CoreML

@available(iOS 13.0, *)
class CameraFrameHandle: UIViewController {
    
    static let rdtModel = rdt();
    static var currentImageFrame: UIImage = UIImage.init() ;
    static let imageMetadataCustom = [
        "UUID": UUID().uuidString,
        "Quality_parameters": [
            "brightness":"10"
        ],
        "RDT_Type":"Flu_Audere",
        "Include_Proof": "True",
    ] as [String : Any];
    
    
    public static func handleFrame(imageFrame: UIImage){
        //currentImageFrame = UIImage.init(cgImage: UIImage() as! CGImage);
        currentImageFrame = imageFrame;
        //doPreprocessing(imageFrame: imageFrame);
    }
    
    private static func doPreprocessing(imageFrame: UIImage){
        /**
         rotate image
         convert to grayscale
         resize to 320*180
         */
        processROIDetection(imageFrame: imageFrame);
    }
    private static func processROIDetection(imageFrame: UIImage){
        guard let rdtImageResizedBytes = imageFrame.jpegData(compressionQuality: 1.0) else { return }
        guard let mlArray = try? MLMultiArray(shape:[1,180,320,1],
                                                   dataType:MLMultiArrayDataType.int32) else {
                                                    fatalError("MLMultiArray conversion error for image")
        }
        
        for index in 0..<rdtImageResizedBytes.count {
            let px: Double = Double(rdtImageResizedBytes[index])
            //mlArray[index] = NSNumber(value: (px/127.5) - 1);
            mlArray[index] = NSNumber(value: px);
        }
        guard let predResult = try? rdtModel.prediction(input_1: mlArray) else {
            fatalError("rdt model error")
        }
        doPostProcessing(roiDetectionResult: predResult.Identity);
    }
    
    private static func doPostProcessing(roiDetectionResult: MLMultiArray){
        // do post processing here
        
    }
    
    public static func evaluateRDTRestApi(callback: @escaping (_ res:String, _ img:String) -> ()){
        // call this on button click from UI
        let paramName = Utils.dict2str(dict: imageMetadataCustom);
        let imageName = "img.jpg";
        print("button evaluateRDTRestApi 0");
//        currentImageFrame = currentImageFrame.rotate(radians: .pi/2)! // Rotate 90 degrees
//        currentImageFrame = Utils.resizeImage(image: currentImageFrame, newWidth: 1280, newHeight: 720)!
//        print("button evaluateRDTRestApi 1");
        let image = currentImageFrame.jpegData(compressionQuality: 1.0)!;
//        print("button evaluateRDTRestApi 2");
        
        HTTPHandlers.API_POST_FORM_DATA(paramName: paramName, imageData: image, fileName: imageName){(result:String, img:String) in
            print(result);
            callback(result,img);
        }
        
//        HTTPHandlers.uploadImage(paramName: paramName, fileName: imageName, image: currentImageFrame){ (result) in
//            print(result);
//            callback(result);
//        }
    }
}

extension Date {
    static var currentTimeStamp: Int64{
        return Int64(Date().timeIntervalSince1970 * 1000)
    }
}

extension UIImage {
    var noir: UIImage? { // grayscale effect
        let context = CIContext(options: nil)
        guard let currentFilter = CIFilter(name: "CIPhotoEffectNoir") else { return nil }
        currentFilter.setValue(CIImage(image: self), forKey: kCIInputImageKey)
        if let output = currentFilter.outputImage,
            let cgImage = context.createCGImage(output, from: output.extent) {
            return UIImage(cgImage: cgImage, scale: scale, orientation: imageOrientation)
        }
        return nil
    }
    func rotate(radians: Float) -> UIImage? {
        var newSize = CGRect(origin: CGPoint.zero, size: self.size).applying(CGAffineTransform(rotationAngle: CGFloat(radians))).size
        // Trim off the extremely small float value to prevent core graphics from rounding it up
        newSize.width = floor(newSize.width)
        newSize.height = floor(newSize.height)

        UIGraphicsBeginImageContextWithOptions(newSize, false, self.scale)
        let context = UIGraphicsGetCurrentContext()!

        // Move origin to middle
        context.translateBy(x: newSize.width/2, y: newSize.height/2)
        // Rotate around middle
        context.rotate(by: CGFloat(radians))
        // Draw the image at its center
        self.draw(in: CGRect(x: -self.size.width/2, y: -self.size.height/2, width: self.size.width, height: self.size.height))

        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage
    }
}
