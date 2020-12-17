import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";

const React = require("react");





class DecimerSpinner extends React.Component {


    constructor(props) {
        super(props);
        this.size = this.props.size || "4x";
    }

    render() {
        return(
            <Container>
                <Row className="justify-content-center">
                    <FontAwesomeIcon icon="robot"  spin/>
                </Row>
            </Container>
        );
    }
}
export default DecimerSpinner;
