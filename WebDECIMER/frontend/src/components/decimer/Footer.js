import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import CheminfLogo from "../cheminf_logo.png";
import Image from "react-bootstrap/Image";
import Navbar from "react-bootstrap/Navbar";
import StickyFooter from 'react-sticky-footer';
import {Link} from "react-router-dom";


import './Footer.css';


const React = require("react");


class Footer extends React.Component {
    render() {
        //className="sticky-bottom border-top"
        const sty ={backgroundColor: '#FFFFFF'};

        return (
            
             <Container  style={sty}  >
                
            

                    <Row className="align-items-center">

                        <Col sm={1} className="align-content-center">
                            <a href="https://cheminf.uni-jena.de/" target="_blank" rel="noreferrer"><Image id="headerIcon" alt="Cheminf logo" className="img-fluid" src={CheminfLogo}/></a>
                        </Col>

                        <Col sm={11} className="align-justify-self-end">

                            <p style={{textAlign: "justify"}}>Deep Learning for Chemical Image Recognition (DECIMER) is a step towards automated chemical image segmentation and recognition.
                             Read mode about DECIMER and how to cite it <a target="_blank" rel="noopener noreferrer"  href="https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00469-w">here</a>. Please submit bug reports, feature requests and general issues through <a target="_blank" rel="noopener noreferrer"  href="https://github.com/Kohulan/DECIMER-Image-Segmentation">the issues tracker at GitHub</a>.
                                DECIMER is actively developed and maintained by the <a target="_blank" rel="noopener noreferrer"  href="https://cheminf.uni-jena.de">Steinbeck group</a> at the University Friedrich-Schiller in Jena, Germany.
                                Copyright &copy; CC-BY-SA 2020</p>

                        </Col>
                    </Row>
                
            </Container>
        
        )
    }
}

export default Footer;