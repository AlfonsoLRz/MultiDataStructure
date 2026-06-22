#include "stdafx.h"
#include "CudaHelper.h"

//

int CudaHelper::_selectedDevice = 0;

// Public methods

CudaHelper::CudaHelper()
{
}

CudaHelper::~CudaHelper()
{
}

size_t CudaHelper::getMaxThreadsBlock()
{
    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, _selectedDevice));

    return prop.maxThreadsPerBlock;
}

void CudaHelper::setDevice(uint8_t deviceIndex)
{
    int numDevices;
    _selectedDevice = 0;
    checkError(cudaGetDeviceCount(&numDevices));

    if (deviceIndex == UINT8_MAX)
    {
        size_t bestScore = 0;
        for (int deviceIdx = 0; deviceIdx < numDevices; deviceIdx++)
        {
            int clockRate;
            int numProcessors;
            checkError(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, deviceIdx));
            checkError(cudaDeviceGetAttribute(&numProcessors, cudaDevAttrMultiProcessorCount, deviceIdx));

            size_t score = clockRate * numProcessors;
            if (score > bestScore)
            {
                _selectedDevice = deviceIdx;
                bestScore = score;
            }
        }

        if (bestScore == 0)
            throw std::runtime_error("CudaModule: No appropriate CUDA device found!");
    }
    else
    {
        _selectedDevice = glm::clamp(deviceIndex, static_cast<uint8_t>(0), static_cast<uint8_t>(numDevices));
    }

    checkError(cudaSetDevice(_selectedDevice));
}

void CudaHelper::synchronize(const std::string& kernelName)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cerr << "CUDA error in " << kernelName << ": " << cudaGetErrorString(error) << '\n';
    CudaHelper::checkError(cudaDeviceSynchronize());
}

void CudaHelper::startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    checkError(cudaEventCreate(&startEvent));
    checkError(cudaEventCreate(&stopEvent));
    checkError(cudaEventRecord(startEvent, 0));
}

float CudaHelper::stopTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    float ms;
    checkError(cudaEventRecord(stopEvent, 0));
    checkError(cudaEventSynchronize(stopEvent));
    checkError(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    return ms;
}

// Protected methods

void CudaHelper::checkError(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}