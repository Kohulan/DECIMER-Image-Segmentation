module.exports = {

  entry: {
        index: "./src/index.js"
    },
    
    output: {
        path: __dirname,
        filename: "./static/frontend/main.js"
    },
    resolve: {
      fallback: {
        util: require.resolve("util/"),
        https: require.resolve("https-browserify"),
        os: require.resolve("os-browserify/browser"),
        http: require.resolve("stream-http"),
        crypto: require.resolve("crypto-browserify"),
        zlib: require.resolve("browserify-zlib"),
        constants: require.resolve("constants-browserify"),
      }
  },
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: {
            loader: "babel-loader"
          }
        },
        
        {
            test: /\.(png|jp(e*)g|svg|gif)$/,
            exclude: /node_modules/,
            use: {
                loader: "url-loader"
            }
        },
        {
            test: /\.(pdf)$/,
            exclude: /node_modules/,
            use: {
                loader: "file-loader"
            }
        },
        {
            test: /\.css$/,
            use: ['style-loader', 'css-loader']
          },

          {
            test: /\.(woff(2)?|ttf|eot|svg)(\?v=\d+\.\d+\.\d+)?$/,
            use: {
              loader: "url-loader"
          }
        } 
      ]
    }
  };
