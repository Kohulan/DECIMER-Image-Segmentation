import React, { Component } from 'react';
import './Decimer.css';
import { Row, Container} from 'react-bootstrap';
import Image from "react-bootstrap/Image";
import DecimerLogo from '../DECIMERlogo.gif';
import Footer from "./Footer";
import {Link} from "react-router-dom";




class About extends React.Component {

    render() {

        return (
            <Container className="align-content-centers">
                <Link to="/"><Image id="headerIcon" alt="DECIMER Logo" className="justify-content-center mb-5" src={DecimerLogo} width={"60%"}></Image></Link>

                <Row className="justify-content-center">
                    <a href="https://cheminf.uni-jena.de/" target="_blank">
                        <Image src="https://cheminf.uni-jena.de/wp-content/uploads/2017/12/cropped-Title_dec_2017.png" fluid/>
                    </a>
                </Row>
                <br/>
                <Row>
                some text about DECIMER here
                </Row>

                <Row className="justify-content-center">
                    <p>For further information visit the <a href="https://cheminf.uni-jena.de/" target="_blank"><i>Cheminformatics and Computational Metabolomics</i> homepage</a>.</p>
                </Row>
                <Row className="fixed-bottom border-top align-content-center text-muted mt-5">
                    <Footer/>
                </Row>

            </Container>
        )


    }
}

export default About;
