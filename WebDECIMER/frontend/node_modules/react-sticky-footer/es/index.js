var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

import React, { Component } from "react";
import PropTypes from "prop-types";

var StickyFooter = function (_Component) {
  _inherits(StickyFooter, _Component);

  function StickyFooter(props) {
    _classCallCheck(this, StickyFooter);

    var _this = _possibleConstructorReturn(this, _Component.call(this, props));

    _this.determineState = function () {
      var scrollOffset = window.pageYOffset + window.innerHeight;
      var contentHeight = document.body.clientHeight - _this.props.bottomThreshold;

      if (!_this.state.isAtBottom && scrollOffset >= contentHeight) {
        _this.setState({ isAtBottom: true });
        _this.props.onFooterStateChange && _this.props.onFooterStateChange(true);
      } else if (_this.state.isAtBottom && scrollOffset < contentHeight - contentHeight * _this.props.stickAtMultiplier) {
        _this.setState({ isAtBottom: false });
        _this.props.onFooterStateChange && _this.props.onFooterStateChange(false);
      }
    };

    _this.handleScroll = function () {
      _this.determineState();
    };

    _this.state = {
      isAtBottom: false
    };
    return _this;
  }

  StickyFooter.prototype.componentDidMount = function componentDidMount() {
    var _this2 = this;

    this.observer = new MutationObserver(function (mutations) {
      var targetHeight = mutations[mutations.length - 1].target.clientHeight;
      var remainingHeight = document.body.clientHeight - targetHeight;
      var totalContentHeight = targetHeight + remainingHeight;

      if (totalContentHeight > window.innerHeight) {
        _this2.determineState();
      } else {
        _this2.setState({ isAtBottom: true });
        _this2.props.onFooterStateChange && _this2.props.onFooterStateChange(true);
      }
    });
    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true
    });
    window.addEventListener("scroll", this.handleScroll);
    this.determineState();
  };

  StickyFooter.prototype.componentWillUnmount = function componentWillUnmount() {
    this.observer.disconnect();
    window.removeEventListener("scroll", this.handleScroll);
  };

  StickyFooter.prototype.render = function render() {
    var fixedStyles = _extends({}, this.props.stickyStyles, {
      position: "fixed",
      bottom: 0
    });
    var initialStyles = _extends({}, this.props.normalStyles, { position: "static" });
    return React.createElement(
      "div",
      null,
      React.createElement(
        "div",
        { style: initialStyles },
        this.props.children
      ),
      !this.state.isAtBottom && React.createElement(
        "div",
        { style: fixedStyles },
        this.props.children
      )
    );
  };

  return StickyFooter;
}(Component);

export { StickyFooter as default };


StickyFooter.propTypes = process.env.NODE_ENV !== "production" ? {
  /**
   * A value that tells the component how close to the bottom should the scroller be before the sticky footer hides
   * and displays at the end of your content. The default is 0, meaning the user needs to scroll all the way to the bottom
   * before the footer hides. A number greater than 0 would cause the sticky footer to hide at some point before the user
   * has scrolled all the way down, depending on the value of the number.
   */
  bottomThreshold: PropTypes.number,
  /**
   * A value that tells the component how much the user should scroll back up before the sticky footer shows up again.
   * The default is 0.001. A number greater than the default would require the user scroll up more before the
   * sticky footer shows up.
   */
  stickAtMultiplier: PropTypes.number,
  /**
   * Styles to be applied to the sticky footer only.
   */
  stickyStyles: PropTypes.object,
  /**
   * Styles to be applied to the footer in its standard location only.
   */
  normalStyles: PropTypes.object,
  /**
   * Callback that informs when the state of the footer has changed from sticky to being in normal document flow, via boolean argument.
   * true means it is in normal flow, false means it is sticky.
   */
  onFooterStateChange: PropTypes.func
} : {};

StickyFooter.defaultProps = {
  bottomThreshold: 0,
  stickAtMultiplier: 0.001,
  stickyStyles: {},
  normalStyles: {}
};