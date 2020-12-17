import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './components/App';
import About from './components/decimer/About';
import { BrowserRouter, Route, Switch } from "react-router-dom";



ReactDOM.render(

  <React.StrictMode>
    <App />
  </React.StrictMode>,
  //document.getElementById('root')
  document.querySelector("#app")
);
