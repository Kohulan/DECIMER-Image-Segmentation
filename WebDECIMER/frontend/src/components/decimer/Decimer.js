import React, { Component } from 'react';
import Dropzone from 'react-dropzone';
import './Decimer.css';
import {Alert, Button, Spinner, Row, Container, Form, OverlayTrigger, Tooltip} from 'react-bootstrap';
import axios from 'axios';

import Image from "react-bootstrap/Image";

import ImageCardBrowser from "./ImageCardBrowser";

//import DecimerLogo from '../DECIMERlogo.png';
import DecimerLogo from '../DECIMERlogo.gif';

import Footer from "./Footer";
import DecimerSpinner from "./DecimerSpinner";

import { faDharmachakra } from "@fortawesome/free-solid-svg-icons";

import { faDownload } from "@fortawesome/free-solid-svg-icons";
import 'font-awesome/css/font-awesome.min.css';


import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";


import { saveAs } from 'file-saver';
import JSZipUtils from 'jszip-utils';

import mdpi_id_list from './mdpi_molecules_ids_2020.js';

import regeneratorRuntime from "regenerator-runtime";

/*

import { attachCookies } from 'superagent';
import dataUriToBuffer from 'data-uri-to-buffer';
*/






class Decimer extends Component {
    constructor(props) {
        super(props);

        this.state = { 
            
            files : [],
            images: [],
            isLoading: false,
            recentArticle:null,
            noImagesInArticle:false,
            retrievedImages: [],
            lastRetrievedImages: [],
            ajaxLoaded: false,
            processingState: 0,
            processingStep:0,
            myIntervals : [],
            luckyURL:"",
            isLucky:false,
        }

        this.updateRef = React.createRef();

        //console.log(mdpi_id_list);

        
    }


    componentDidUpdate(prevProps, prevState, snapshot) {


        if(this.state.isLoading){
            this.scrollToRef(this.updateRef);
        }else if(this.state.ajaxLoaded){
            this.scrollToRef(this.updateRef);
        }
    }

    scrollToRef(ref) {
        window.scrollTo(0, ref.current.offsetTop);
    }

    renderLuckyTooltip = (props) => (
        <Tooltip id="button-tooltip" {...props}>
          Download and segment a random recent article from <a href="https://www.mdpi.com/journal/molecules" target="_blank" rel="noreferrer">MDPI Molecules</a>
        </Tooltip>
      );

     onDrop = (files) => {
         console.log("onDrop:");
         console.log(files);
        this.setState({
            isLoading:true,
            files:[],
            recentArticle:null,
            retrievedImages:[],
            ajaxLoaded: false,
            luckyURL:"",
            isLucky:false,
        });
        this.loadArticle(files);
     }

     loadArticle=(files)=>{
        setTimeout( () => {
            this.setState({
                files,
                isLoading:false
            },
            () => {
                console.log(this.state.files);
            }
            );
        }, 0);
        
     }



    handleImageDownload = () => {

        var JSZip = require("jszip");
        var zip = new JSZip();
        var count = 0;
        var zipFilename = "DECIMER_segmented_images_from_"+this.state.lastRetrievedImages[0].ori_article_name+".zip";

        var totalImagesToZip = this.state.lastRetrievedImages.length*2 ;

        this.state.lastRetrievedImages.forEach(function(segImage){

            let tmp = segImage.clean_image.split("/");
            let filename_clean_image = tmp[tmp.length-1].replace(".png_", " image ");
            
            tmp = segImage.bnw_image.split("/");
            let filename_bnw_image = tmp[tmp.length-1].replace(".png_", " image ");

            var httpVersion_clean = segImage.clean_image.replace("http", "https");
            var httpVersion_bnw = segImage.bnw_image.replace("http", "https");

            JSZipUtils.getBinaryContent(httpVersion_clean, function (err, data) {
                if(err) {
                   throw err; 
                }
                zip.file(filename_clean_image, data, {binary:true});
                count++;
                if (count === totalImagesToZip) {

                  zip.generateAsync({type:'blob'}).then(function(content) {
                    saveAs(content, zipFilename);
                 });
                }
             });


             JSZipUtils.getBinaryContent(httpVersion_bnw, function (err, data) {
                if(err) {
                   throw err; 
                }
                zip.file(filename_bnw_image, data, {binary:true});
                count++;
                if (count === totalImagesToZip) {

                  zip.generateAsync({type:'blob'}).then(function(content) {
                    saveAs(content, zipFilename);
                 });
                }
             });
        });


    }


    feelingLucky = async event =>{



        //update default params
        this.setState({
            isLoading:true,
            files:[],
            recentArticle:null,
            retrievedImages:[],
            ajaxLoaded: false,
            luckyURL:"",
            isLucky:true,
        });


        var random_id = mdpi_id_list[Math.floor(Math.random()*mdpi_id_list.length)];

        //getting a random paper ID


        var proxyUrl = 'https://cors-anywhere.herokuapp.com/',
        targetUrl = "https://mdpi.com"+random_id,
        luckyUrl = "https://mdpi.com"+random_id;
        luckyUrl = luckyUrl.replace("/pdf","");
        this.setState({
            luckyURL:luckyUrl
        });

        const method = 'GET';
        const url = proxyUrl + targetUrl;

    
        let blob = await fetch(url).then(r => r.blob());

        blob.lastModifiedDate = new Date();
        blob.name = "mdpi_molecules_lucky.pdf";

        this.setState({
                files:[blob],
                isLoading:false,
                luckyURL:targetUrl,
                

                
            });

  

            this.sendArticle();

    }



     
     sendArticle = () =>{

            console.log("posting article");
            console.log(this.state.files[0].name);


            let formData = new FormData();
            formData.append('article', this.state.files[0], this.state.files[0].name);
            
            
            this.setState({
                isLoading:true,
                ajaxLoaded:false
            });

            axios.post(
                '/api/segmentation/uploaded/', 
                formData,
                {   
                    timeout:2400000,
                    headers:{
                        'accept':'application/json',
                        'content-type':'multipart/form-data',
                    }
                }
            ).then(resp=>{
                this.setState({
                    isLoading:false,
                    files:[]
                });
                console.log("file send and received");
                console.log(resp.data.id);
                console.log(resp.data)

                //this.getArticleClass(resp);
                this.getSegmentedImages(resp.data);

            }).catch(err=>{


                if(err.response && (err.response.status==504 || err.response.status==502 )){


                    //Checking on the segmentation state, as the server times out

                    var myTimer = setInterval(this.checkStateUntilGood, 10000);
                    this.state.myIntervals.push(myTimer);

                    if(this.state.ajaxLoaded || this.state.processingState===2 || this.state.processingState===-1 || this.state.files.length==0){
                        clearInterval(myTimer);
                    }
                    

                }else{
                    this.setState({
                        ajaxLoaded:false,
                        isLoading:false
                    });
                    console.log(err);
                    alert("Something went wrong on our post side, please retry!")
                    //TODO add alert here!
                }
            });

            var stepTimer = setInterval(this.getProcessingStep, 2000);
            this.state.myIntervals.push(stepTimer);

                    if(this.state.ajaxLoaded || this.state.processingStep==4 || this.state.files.length==0){
                        clearInterval(stepTimer);
                    }
        
        }




     checkStateUntilGood = () =>{

        //need to GET by ori_name + max_id
                console.log("check article state until good");



                axios.get(`/api/segmentation/uploaded/?ori_name=${this.state.files[0].name}`,{
                    headers:{
                        'accept':'application/json'
                    }
                }).then(resp=>{
                    
                    
                    console.log(resp.data);// this is a list

                    let latest_id = 0;

                    for(var i; i< resp.data.length; i++){
                        if(resp.data[i].id>latest_id){
                            latest_id=resp.data[i].id;
                        }
                    }

                    console.log("received processing state:");
                    console.log(resp.data[latest_id].processingState);

                    if(resp.data[latest_id].processingState ===1){
                        //TODO wait 5 sec
                        
                        console.log("still processing");
                        this.setState({
                            processingState: resp.data[latest_id].processingState,
                        });

                    }else if(resp.data[latest_id].processingState === 2){
 
                        //get images with the id
                        this.setState({
                            processingState: resp.data[latest_id].processingState,
                            isLoading:false,
                            files:[]
                        });
                        console.log("file send and finally processed!");
                        console.log(resp.data.id);
                        console.log(resp.data)
        
                        //this.getArticleClass(resp);
                        this.getSegmentedImages(resp.data[latest_id]);

                    }else if(resp.data[latest_id].processingState === -1){
                        
                        //set returned images to zero
                        this.setState({
                            processingState: resp.data[latest_id].processingState,
                            isLoading:false,
                            ajaxLoaded: true,
                            files:[],
                            retrievedImages: [],
  
                        });

                    }

                }).catch(err=>{
                    this.setState({
                        ajaxLoaded:false,
                        isLoading:false
                    });
                    console.log(err);
                    console.log("probably couldn't find the article by name!");
                    alert("Something went wrong on our get side, please retry!")

                });
    }


    getProcessingStep = () =>{
        console.log("retrieving the processing step");

        axios.get(`/api/segmentation/uploaded/?ori_name=${this.state.files[0].name}`,{
            headers:{
                'accept':'application/json'
            }
        }).then(resp=>{
            
            console.log("received something");
            let latest_id = 0;

            for(var i; i< resp.data.length; i++){
                if(resp.data[i].id>latest_id){
                    latest_id=resp.data[i].id;
                }
            }

            console.log("received processing state:");
            console.log(resp.data[latest_id].processingStep);

            // TODO check the step and adapt

            if(resp.data[latest_id].processingStep ===0){
                this.setState({
                    processingStep:0
                });

            }else if(resp.data[latest_id].processingStep ===1){
                this.setState({
                    processingStep:1
                });
            }else if(resp.data[latest_id].processingStep ===2){
                this.setState({
                    processingStep:2
                });
            }else if(resp.data[latest_id].processingStep ===3){
                this.setState({
                    processingStep:3
                });
            }else if(resp.data[latest_id].processingStep ===4){
                this.setState({
                    processingStep:4
                });

            }

        }).catch(err=>{
            this.setState({
                ajaxLoaded:false,
                isLoading:false
            });
            console.log(err);
            console.log("probably couldn't find the article by name!");
            alert("Something went wrong on our get side, please retry!")

        });
    }




     getSegmentedImages = (obj) =>{
         console.log("trying to get segmented images");
         const httpClient = axios.create();
        httpClient.defaults.timeout = 2400000;

        httpClient.get(`/api/segmentation/segmented/?ori_article_id=${obj.id}`, 
        {
            headers:{
                'accept':'application/json',
            }
        }
    ).then(resp=>{

        console.log("catched images");
        console.log(resp);

        console.log("clearing intervals");
        this.state.myIntervals.forEach(function(interval){
            clearInterval(interval);
        });
    

        this.setState({
            ajaxLoaded: true,
            retrievedImages:resp.data,
            lastRetrievedImages : [...resp.data],
            myIntervals:[],
            
        });


        

    }).catch(err=>{
        this.setState({
            ajaxLoaded:false,
            isLoading:false
        });
        console.log(err);
        alert("Something went wrong on our get side, please retry!")
        //TODO add alert here!
    })

     }



    
    render() { 


        const files = this.state.files.map(file => (
            <p className='text-muted'>
              {file.name} - {file.size} bytes
            </p>
          ));



        return ( 

            <Container className="align-content-centers">

                    <Image id="headerIcon" alt="DECIMER Logo" className="justify-content-center mb-5" src={DecimerLogo} width={"60%"}></Image>

                    <Container className="align-content-centers flex">
                        <Dropzone id="decimerDropzone" onDrop={this.onDrop} accept='application/pdf'>
                            {({isDragActive, getRootProps, getInputProps}) => (
                            <section className="container">
                                <div {...getRootProps({className: 'dropzone back'})}>
                                <input {...getInputProps()} />
                                
                                <p className='text-muted'>{isDragActive ? "Drop the file" : "Drop PDF files here, or click to select files"}</p>
                                </div>
                                <aside>
                                {files}
                                </aside>
                                
                                <Row className="justify-content-center">
                                    <Button variant='info' size='sm' className='mt-3 mr-2' onClick={this.sendArticle}>Segment</Button>

                                    <OverlayTrigger
                                        placement="right"
                                        delay={{ show: 0, hide: 777 }}
                                        overlay={this.renderLuckyTooltip}
                                        variant='secondary'
                                    >
                                        <Button variant='info' size='sm'  className='mt-3 ml-2' onClick={this.feelingLucky} >I'm feeling lucky</Button>
                                    </OverlayTrigger>
                                </Row>

                                {this.state.isLucky && this.state.luckyURL &&
                                <Alert variant='info' className="mt-5" ref={this.updateRef}>
                                    Extracting molecular images from <a href={this.state.luckyURL} target="_blank" rel="noreferrer">this article</a> from MDPI metabolites
                                </Alert>
                                }

                                {this.state.isLoading &&
                                <Container className="align-content-centers flex mt-2">
                                    <Row className="justify-content-center mt-2" ref={this.updateRef}>
                                        <FontAwesomeIcon icon={faDharmachakra} className="standAloneIcon" size={"4x"} variant='warning' spin/>
                                    </Row>
                                    <Row  className="justify-content-center mt-2">
                                        <p className='text-muted'>Image segmentation started, it might take time...</p>
                                    </Row>
                                    {this.state.processingStep===1 &&
                                    <Row className="justify-content-center mt-2" ><p className='text-muted'>PDF uploaded... saving...</p></Row>}
                                    {this.state.processingStep===2 &&
                                    <Row className="justify-content-center mt-2" ><p className='text-muted'>PDF cut in pages and converted to JPEG format...</p></Row>}
                                    {this.state.processingStep===3 &&
                                    <Row className="justify-content-center mt-2" ><p className='text-muted'>Running mask expansion: molecule detection and search for all atoms. AI at work!</p></Row>}
                                    {this.state.processingStep===4 &&
                                    <Row className="justify-content-center mt-2" ><p className='text-muted'>Cleaning detected images.....</p></Row>}
                                </Container>
                                }

                                {/* {this.state.noImagesInArticle &&
                                <Alert variant='danger'>
                                    <span className="sr-only">No molecular image was detected in the article!</span>
                                    <p>This can be a problem on our side, please try refreshing the page and resubmit the article</p>
                                </Alert>
                                } */}

                                {(!this.state.isLoading && this.state.ajaxLoaded) &&
                                <React.Fragment>
                                    <Alert variant='primary' className="mt-5" ref={this.updateRef}>
                                        Found {this.state.retrievedImages.length} image(s) of molecules in the submitted article.
                                    </Alert>
                                    <Row className="justify-content-center mt-2 mb-2">

                                        <Button id="dlImages" variant="success" size="sm" onClick={this.handleImageDownload}>
                                            <FontAwesomeIcon icon={faDownload} fixedWidth/>
                                            &nbsp;Download segmented images
                                        </Button>
                                    </Row>
                                    <Row>
                                            <ImageCardBrowser retrievedImages={this.state.retrievedImages}/>
                                    </Row>
                                </React.Fragment>
                                }

                            </section>
                            )}
                        </Dropzone>
                        </Container>
                        <Container className="align-content-centers">
                        <div class="phantom"></div>
                        </Container>
            

                <Row className="fixed-bottom border-top align-content-center text-muted mt-5">
                    <Footer/>
                </Row>
            </Container>
         );
    }
}
 
export default Decimer;