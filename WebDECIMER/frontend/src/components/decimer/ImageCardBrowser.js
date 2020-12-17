import React, { Component } from 'react';

import CardDeck from "react-bootstrap/CardDeck";
import Card from "react-bootstrap/Card";

import Container from "react-bootstrap/Container";

import SegmentedImageCardItem from "./SegmentedImageCardItem";


export default class ImageCardBrowser extends Component {

    
    render() { 

        console.log("printing props:");
        console.log(this.props);

        const cardRowSize = 4;

        //let path_to_images = this.props.segmentedArticle.path_to_segmented_images;
        
        let retrievedImages = this.props.retrievedImages;

        console.log(retrievedImages);

        //let image_paths_list = segmentedArticle.all_segmented_images_names.split("$x$x$x$");
        //$x$x$x$

        //console.log(image_paths_list);


        let emptyCardKey = 0;
        let cardRows = [];

        /*let isOdd = false;
        if(Math.abs(retrievedImages.length % 2) == 1){
            //if the number of found images is odd
            isOdd=true;
        }*/

        while(retrievedImages.length > 0){
            let cardRow = [];

            retrievedImages.splice(0, cardRowSize).map(segImage => {
                if(typeof segImage !== 'undefined'){
                    cardRow.push(
                        <SegmentedImageCardItem key={segImage.id+"a"} segImage = {segImage.clean_image}/>
                    );
                    /*cardRow.push(
                        <SegmentedImageCardItem key={segImage.id+"b"} segImage = {segImage.bnw_image}/>
                    );*/
                }
            });
            
            /*if(isOdd){
                while (cardRow.length < cardRowSize+2) {
                    cardRow.push(
                        <Card key={emptyCardKey++} style={{visibility: "hidden"}}>
                            <Card.Body>
                                <Card.Text>empty</Card.Text>
                            </Card.Body>
                        </Card>
                    );
                }
         }*/
         while (cardRow.length < cardRowSize) {
            cardRow.push(
                <Card key={emptyCardKey++} style={{visibility: "hidden"}}>
                    <Card.Body>
                        <Card.Text>empty</Card.Text>
                    </Card.Body>
                </Card>
            );
        }

            cardRows.push(<CardDeck key={cardRows.length}>{cardRow}</CardDeck>);
        }



        return ( <Container>{cardRows}</Container> );
    }
}
 
