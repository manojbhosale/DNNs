import os

IMG_SIZE = 24
NUM_CLASSES = 10
EXAMPLES_PER_EPOCH_TRAIN = 50000
EXAMPLES_PER_EPOCH_EVAL = 10000

def readCifer10(fileNameQueue):
    class CIFAR10record(Object):
        pass
    result = CIFAR10record()
    
    labelBytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    imageBytes = result.height + result.width + result.depth
    recordBytes = imageBytes + labelBytes
    reader = tf.FixedLengthRecordReader(record_bytes=recordBytes)
    result.key, value = reader.read(fileNameQueue)
    
    recordBytes = tf.decode_raw(value, tf.uint8)
    
    result.label = tf.cast(tf.strided_slice(recordBytes, [0], [labelBytes]), tf.int32)
    depthMajor = tf.reshape(tf.strided_slice(recordBytes,[labelBytes],[labelBytes+ imageBytes]),[result.depth, result.height, result.width])
    result.uint8image
    
    
    
    

def createInput(evalData, dataDir, batchSize):
    if not evalData:
        fileNames = [os.path.join(dataDir, "data_batch_%d.bin" % i) for i in xrange(1,6)]
        numExamplesPerEpoch = EXAMPLES_PER_EPOCH_TRAIN
    else:
        fileNames = [os.path.join(datDir, 'test_batch.bin')]
        numExamplesPerEpoch = EXAMPLES_PER_EPOCH_EVAL
    
    for f in fileNames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: '+ f)
    
    fileNameQueue = tf.train.string_input_producer(fileNames)
    
    readInput = readCifar10(fileNameQueue)
    reshapedImage = tf.cast(readInput.uint8image, tf.float32)
    
    height = IMG_SIZE
    width = IMG_SIZE
    
    resizedImage = tf.image.resize_image_with_crop_or_pad(reshapedImage, height, width)
    
    floatImage = tf.image.per_image_standardization(resizedImage)
    floatImage.set_shape([height, width, 3])
    readInput.label.input.set_shape([1])
        
    minFractExampleQueue = 0.4
    minQueueExamples = int(minFractExampleQueue * numExamplesPerEpoch)
    
    return generatemgLabelBatch(floatImage, readInput.label, minQueueExamples, batchSize, shuffle=False)
    
