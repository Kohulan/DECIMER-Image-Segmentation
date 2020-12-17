import React, { Component } from 'react';
import Card from "react-bootstrap/Card";

import Image from "react-bootstrap/Image";

import './Decimer.css';



export default class SegmentedImageCardItem extends Component {
    state = {  }
    render() { 

        console.log("in card item");
        console.log(this.props);
        let segImage = this.props.segImage;

        let tmp = segImage.split("/");
        let imageTitle = tmp[tmp.length-1].replace(".png_", " image ");
        imageTitle = imageTitle.replace(/.png|_/g, " ");
        imageTitle = imageTitle.trim();
        console.log("$"+imageTitle+"$");
        if(imageTitle.endsWith("clean") || imageTitle.endsWith("bnw")){
            imageTitle = imageTitle.replace("clean", "");
            
        }else{
            var lastIndex = imageTitle.lastIndexOf(" ");//removing the last word if it's a unique addition
            imageTitle = imageTitle.substring(0, lastIndex);
            imageTitle = imageTitle.replace("clean", "");
        }

        return ( 
            <Card className="cardBrowserItem" >
                <Card.Body>
                    <Card.Title>
                        {imageTitle}
                    </Card.Title>
                    <Image className='justify-content-center' src={segImage} width={"90%"}></Image>
                </Card.Body>
            </Card>
         );
    }
}
 
