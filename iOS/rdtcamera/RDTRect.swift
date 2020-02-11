//
//  RDTRect.swift
//  RDT Camera
//
//  Created by Akhil Kumar on 25/10/19.
//  Copyright © 2019 IPRD. All rights reserved.
//

import Foundation
import UIKit

class RDTRect: UIView {
    override init(frame: CGRect) {
        super.init(frame: frame)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(_ rect: CGRect) {
        let h = rect.height
        let w = rect.width
        let color:UIColor = UIColor.red
        
        let drect = CGRect(x: (w * 0.25),y: (h * 0.25),width: (w * 0.5),height: (h * 0.5))
        let bpath:UIBezierPath = UIBezierPath(rect: drect)
        
        color.set()
        bpath.stroke()
    }
    
    func setProps(_ resolution: CGSize, _ position:CGPoint ) {
        frame.size.width = resolution.width
        frame.size.height = resolution.height
        move(position);
    }
    
    func move(_ position:CGPoint) {
        frame.origin = position
    }
}
