const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const { promisify } = require('util');
const fs = require('fs');
const readFile = promisify(fs.readFile);
const path = require('path');

async function loadImage(imageName) {
    console.log('loading image');
    const imagesShape = [1, 28, 28, 3];
    const image = await readFile(`${__dirname}/${imageName}.jpeg`);
    const imageBuffer = await sharp(image).resize(28, 28).toBuffer();
    const allImages = new Float32Array(tf.util.sizeFromShape(imagesShape));
    allImages.set(imageBuffer, 0);
    return tf.tensor4d(allImages, imagesShape)
}

async function loadModel() {
    // Get reference to bundled model assets
    // const modelJson = require('./practicemnist-node/model.json');
    // const modelWeights = require('./practicemnist-node/weights.bin');
    return await tf.loadLayersModel(`file://${path.resolve(__dirname, 'model.json')}`, {
        weightPathPrefix: `file://${path.resolve(__dirname, 'weights.bin')}`
    });
}

async function classify(model, imageName) {
    if(!model) {
        model = await loadModel();
    }
    const imageTensor = await loadImage(imageName);
    const prediction = model.predict(imageTensor);
    console.log('-------------------------------');
    console.log('prediction for image', imageName)
    console.log(prediction);
    model.predict(imageTensor).print();
    console.log('--------------------------------');
}
if(process.argv[2]) {
    classify(null, process.argv[2])
}
module.exports = classify;