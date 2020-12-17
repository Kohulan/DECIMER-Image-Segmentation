# react-sticky-footer

A simple sticky footer for React

## How to install
`npm install react-sticky-footer --save` or `yarn add react-sticky-footer`

## What Sticky Footer does?

The sticky footer will stick to the bottom of the browser, on pages with long content. Once the user scrolls to the bottom the sticky footer will disappear, and will display the footer at the end of the content (in relation to where the StickyFooter tag was placed in your document).

If the content size changes without a scroll, the component will auto-refresh its state to determine if it should display the sticky footer or not.

The component will depend on the document body for height and mutation checks.

## What Sticky Footer doesn't do

On content shorter than your browser's height, the sticky footer will render below the content, and will not stick to the bottom. In the future I may add an option to stick to the browser in these cases.

## How do I use it?
```js
import StickyFooter from 'react-sticky-footer';
```

```jsx
<StickyFooter
    bottomThreshold={50}
    normalStyles={{
    backgroundColor: "#999999",
    padding: "2rem"
    }}
    stickyStyles={{
    backgroundColor: "rgba(255,255,255,.8)",
    padding: "2rem"
    }}
>
    Add any footer markup here
</StickyFooter>
```

## How can I control the sticky footer?

### Props

__bottomThreshold__ (optional): A value that tells the component how close to the bottom should the scroller be before the sticky footer hides and displays at the end of your content. The default is 0, meaning the user needs to scroll all the way to the bottom before the footer hides. A number greater than 0 would cause the sticky footer to hide at some point before the user has scrolled all the way down, depending on the value of the number.

__stickAtThreshold__ (optional): A value that tells the component how much the user should scroll back up before the sticky footer shows up again. The default is 0.001. A number greater than the default would require the user scroll up more before the sticky footer shows up.

__stickyStyles__ (optional): Styles to be applied to the sticky footer only.

__normalStyles__ (optional): Styles to be applied to the footer in its standard location only.

__onFooterStateChange__ (optional): Callback that informs when the state of the footer has changed from sticky to being in normal document flow, via boolean argument. true means it is in normal flow, false means it is sticky.
