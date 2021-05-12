const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { promisify } = require('util');
const readDir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const sharp = require('sharp');

class AI {
    compile() {
        const model = tf.sequential();
        //input layer
        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 3],
            kernelSize: [3,3],
            filters: 32,
            activation: 'relu',
            kernelInitializer: 'VarianceScaling'
        }));


        model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));

        model.add(tf.layers.flatten({}));

        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax'
        }));

        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: 'Adam',
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

            const catFiles = await readDir(`${__dirname}/data/Cat/`);
            const dogFiles = await readDir(`${__dirname}/data/Dog/`);
            const j = catFiles.length;
            const k = dogFiles.length;
            const errors = {cats: [], dogs: []};
            for (let i = j; i > 0; i--) {
                try {
                    const image = await readFile(`${__dirname}/data/Cat/${catFiles[i - 1]}`);
                    const resizedBuffer = await sharp(image).resize(28, 28).toBuffer();
                    images.push(tf.node.decodeJpeg(resizedBuffer, 3));
                    labels.push(0);
                } catch (error) {
                    errors.cats.push(catFiles[i - 1]);
                }
            }
            for (let i = k; i > 0; i--) {
                try {
                    const image = await readFile(`${__dirname}/data/Dog/${dogFiles[i - 1]}`);
                    const resizedBuffer = await sharp(image).resize(28, 28).toBuffer();
                    images.push(tf.node.decodeJpeg(resizedBuffer, 3));
                    labels.push(1);
                } catch (error) {
                    errors.dogs.push(dogFiles[i - 1]);
                }
            }
            // const y = tf.oneHot(labels, 2);
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);

            await model.fit(tf.stack(images), ys, {
                epochs: 2
            })
        } catch (error) {
            console.log(error);
        }

    }
}
(async function() {
    const ai = new AI();
    await ai.run();
})();
