const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const readDir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const sharp = require('sharp');

class AI {
    compile() {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 3],
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }));
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }));
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }));
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }));
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dropout({rate: 0.25}));
        model.add(tf.layers.dense({units: 512, activation: 'relu'}));
        model.add(tf.layers.dropout({rate: 0.5}));
        model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
        model.compile({
            optimizer: 'rmsprop',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });
        return model;
    }

    //run model / predict
    async run(){
        try {
            const model = this.compile();

            const images = [];
            const labels = [];

            const imageFiles = await readDir(`${__dirname}/data/`);
            const j = imageFiles.length;
            const errors = [];
            for (let i = 0; i < j; i++) {
                try {
                    const image = await readFile(`${__dirname}/data/${imageFiles[i]}`);
                    const resizedBuffer = await sharp(image).resize(28, 28).toBuffer();
                    images.push(resizedBuffer);
                    labels.push([imageFiles[i].split('_')[1].startsWith('cat') ? 1 : 0]);
                } catch (error) {
                    errors.push(imageFiles[i]);
                }
            }
            console.log(`we are unable to load ${errors.length} files. Ignored`);
            const size = images.length;
            const imagesShape = [size, 28, 28, 3];
            const allImages = new Float32Array(tf.util.sizeFromShape(imagesShape));
            const allLabels = new Int32Array(tf.util.sizeFromShape([size, 1]));

            let imageOffset = 0;
            let labelOffset = 0;
            const imageFlatSize = 28 * 28;
            for (let i = 0; i < size; ++i) {
                allImages.set(images[i], imageOffset);
                allLabels.set(labels[i], labelOffset);
                imageOffset += imageFlatSize;
                labelOffset += 1;
            }

            const xs = tf.tensor4d(allImages, imagesShape)
            const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), 2).toFloat();

            await model.fit(xs, ys, {
                epochs: 10,
                batchSize: 16,
                validationSplit: 0.2
            });

            const classify = require('./test');
            await classify(model, 'cat');
            await classify(model, 'dog');
            await model.save(`file://${path.resolve(__dirname)}`);
        } catch (error) {
            console.log(error);
        }

    }
}
(async function() {
    const ai = new AI();
    await ai.run();
})();
