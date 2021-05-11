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
            inputShape: [250, 250, 3],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));

        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: 2,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        }));

        model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: tf.train.adam(),
            metrics: ['accuracy'],
        });
        return model;
    }

    //run model / predict
    async run(){
        try {
            const model = this.compile();

            const catTensors = [];
            const dogTensors = [];

            const catFiles = await readDir(`${__dirname}/data/Cat/`);
            const dogFiles = await readDir(`${__dirname}/data/Dog/`);
            const j = catFiles.length;
            const k = dogFiles.length;
            const errors = {cats: [], dogs: []};
            for (let i = j; i > 0; i--) {
                try {
                    const image = await readFile(`${__dirname}/data/Cat/${catFiles[i - 1]}`);
                    const resizedBuffer = await sharp(image).resize(250, 250).toBuffer();
                    catTensors.push(tf.node.decodeJpeg(resizedBuffer));
                } catch (error) {
                    errors.cats.push(catFiles[i - 1]);
                }
            }
            for (let i = k; i > 0; i--) {
                try {
                    const image = await readFile(`${__dirname}/data/Dog/${dogFiles[i - 1]}`);
                    const resizedBuffer = await sharp(image).resize(250, 250).toBuffer();
                    dogTensors.push(tf.node.decodeJpeg(resizedBuffer));
                } catch (error) {
                    errors.dogs.push(dogFiles[i - 1]);
                }
            }

            const input = [...catTensors, ...dogTensors];
            const output = [tf.tensor1d([1]), tf.tensor1d([0])];

            await model.fit(input, output, {
                epochs: 3,
                batchSize: 10
            });
        } catch (error) {
            console.log(error);
        }

    }
}
(async function() {
    const ai = new AI();
    await ai.run();
})();
