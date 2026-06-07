import math
import warnings

import numpy as np

from midigpt.attributes.tension._processing import normal_round

# Farbood model defaults — must match constants in the original testTensionModel.py
SAMPLE_RATE = 10
MEMORY_WINDOW_DUR = 3
MEMORY_WEIGHT = 2
SLIDER_ONSET = False
ATTENTIONAL_WINDOW_DUR = 3
WINDOW_SHIFT = 0.25
INIT_SLOPE = 1
LAG = 1
FEATURE_WEIGHTS_PITCHED = [2, 3, 3, 2, 1, 1]
FEATURE_WEIGHTS_DRUMS = [2, 0, 3, 2, 0, 0]


def runModel(
    features, target, featureList, featureWeights,
    memoryWindowDur, sampleRate,
    attentionalWindowDur, windowShift,
    name, memoryWeight, initSlope, lag,
    sliderOnset=False, showFigures=False,
):
    predictionResult = []
    target = np.array(target)
    features = np.array(features)

    error = False
    if len(featureWeights) != len(features):
        warnings.warn("featureWeights length does not match number of feature graphs")
        error = True
    if len(featureWeights) != len(featureList):
        warnings.warn("featureWeights length does not match featureList length")
        error = True
    if target.size > 0 and target.size != len(features[0]):
        warnings.warn(
            f"feature vector lengths ({len(features[0])}) do not match "
            f"target length ({target.size})"
        )
        error = True
    if error:
        return predictionResult

    numPoints = len(features[0])
    numFeatures = len(featureWeights)

    samplesPerAttentionalWindow = normal_round(sampleRate * attentionalWindowDur)
    samplesPerMemoryWindow = normal_round(memoryWindowDur * sampleRate)

    prediction = []
    endReached = False
    shift = normal_round(windowShift * sampleRate)
    if shift == 0:
        shift = 1

    startWindowDur = 2 * sampleRate
    prevSlope = initSlope

    absoluteValsofWeights = [int(math.fabs(ele)) for ele in featureWeights]
    scaleWeightFactor = sum(absoluteValsofWeights)
    weights = np.array(featureWeights) / scaleWeightFactor
    initialSliderMovementSlope = 0.25 / scaleWeightFactor
    memoryMultiplier = memoryWeight
    memoryWindowActive = False

    for i in range(-samplesPerAttentionalWindow + 2, numPoints, shift):
        startpt = i
        endpt = samplesPerAttentionalWindow + i - 1
        x = np.array(range(0, samplesPerAttentionalWindow))

        if numPoints - endpt <= 0:
            endpt = numPoints - 1
            x = np.array(range(0, endpt - startpt + 1))
        elif startpt < 0:
            startpt = 0
            endpt = samplesPerAttentionalWindow + i - 1
            if endpt < 1:
                endpt = 1
            x = np.array(range(0, endpt - startpt + 1))

        if memoryWindowDur > 0:
            memEnd = startpt - 1
            memStart = memEnd - samplesPerMemoryWindow + 1
            if memStart < 0:
                memStart = 0
            memoryWindowActive = memEnd >= 3

        if not endReached and endpt > startpt:
            currFeatures = features[:, startpt:endpt + 1]
            slopes = np.zeros(numFeatures)
            for j in range(numFeatures):
                p = np.polyfit(x, currFeatures[j], 1)
                slopes[j] = p[0]
                if np.isnan(slopes[j]):
                    slopes[j] = 0
                    warnings.warn("NaN in feature data")

            if memoryWindowDur > 0 and memoryWindowActive:
                currMemoryWindowSamples = memEnd - memStart + 1
                xMem = range(0, currMemoryWindowSamples)
                p1 = np.polyfit(xMem, prediction[memStart:memEnd + 1], 1)
                prevSlope = p1[0]
                if np.isnan(prevSlope):
                    prevSlope = 0

            if sliderOnset and i < startWindowDur:
                slopeTotal = initialSliderMovementSlope
            else:
                slopeTotal = sum(weights * slopes)

            if np.isnan(slopeTotal):
                warnings.warn("NaN slopeTotal")

            epsilon = 0.0001
            decay = 0.001

            if memoryWindowDur > 0 and memoryWindowActive:
                if (
                    slopeTotal < epsilon and slopeTotal > -epsilon
                    and prevSlope < epsilon and prevSlope > -epsilon
                ):
                    slopeTotal = slopeTotal - decay
                elif (slopeTotal > 0 and prevSlope > 0) or (slopeTotal < 0 and prevSlope < 0):
                    slopeTotal = slopeTotal * memoryMultiplier

            y = slopeTotal * x

            if startpt == 0:
                prediction = y
            else:
                start = prediction[0:startpt]
                originalStartMergeVal = prediction[startpt]
                middle = np.array(y[0:len(prediction) - startpt] + prediction[startpt:]) / 2

                if middle.size > 0:
                    offset1 = originalStartMergeVal - middle[0]
                    middle = middle + offset1

                endChunk = y[len(middle):]
                if endChunk.size > 0 and middle.size > 0:
                    offset2 = middle[-1] - endChunk[0]
                    endChunk = endChunk + offset2

                prediction = np.concatenate((start, middle, endChunk), axis=0)

        if endpt == numPoints - 1:
            endReached = True

    prediction = (prediction - np.mean(prediction)) / np.std(prediction, ddof=1)

    lagOffset = int(lag * sampleRate)
    if lagOffset > 0:
        trim = np.zeros(lagOffset)
        trim[:] = prediction[0]
        predictionTrimmed = prediction[0:-lagOffset]
        prediction = np.concatenate((trim, predictionTrimmed), axis=0)

    return prediction
