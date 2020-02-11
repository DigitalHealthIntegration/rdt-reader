//
//  Utils.swift
//  RDT Camera
//
//  Created by developer on 24/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

import UIKit

class Utils {
    
    public static func dict2str(dict: Dictionary<String, Any>) -> String {
        var result: String = "";
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: dict, options: [])
            result = String(data: jsonData, encoding: .utf8)!
        } catch {
            print(error.localizedDescription)
        }
        return result;
    }
    
    
    public static func resizeImage(image: UIImage, newWidth: CGFloat, newHeight: CGFloat) -> UIImage? {

        //let scale = newWidth / image.size.width
        //let newHeight = image.size.height * scale
        UIGraphicsBeginImageContext(CGSize(width: newWidth, height: newHeight))
        image.draw(in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))

        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        return newImage
    }
}


