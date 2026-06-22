#pragma once

class CudaHelper
{
protected:
	static int _selectedDevice;

public:
	CudaHelper();
	virtual ~CudaHelper();
	
	static void checkError(cudaError_t result);

	template<typename T>
	static void downloadBufferGPU(T*& bufferPointer, T* buffer, size_t size);

	template<typename T>
	static void free(T*& bufferPointer) { cudaFree(bufferPointer); }

	static size_t getMaxThreadsBlock();
	static size_t getNumBlocks(size_t size, size_t blockThreads) { return (size + blockThreads) / blockThreads; }

	template<typename T>
	static void initializeBufferGPU(T*& bufferPointer, size_t size, T* buffer = nullptr);

	static void setDevice(uint8_t deviceIndex = UINT8_MAX);

	static void synchronize(const std::string& kernelName = "");

	static void startTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);

	static float stopTimer(cudaEvent_t& startEvent, cudaEvent_t& stopEvent);
};

template<typename T>
inline void CudaHelper::downloadBufferGPU(T*& bufferPointer, T* buffer, size_t size)
{
	CudaHelper::checkError(cudaMemcpy(buffer, bufferPointer, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template<typename T>
inline void CudaHelper::initializeBufferGPU(T*& bufferPointer, size_t size, T* buffer)
{
	CudaHelper::checkError(cudaMalloc((void**)&bufferPointer, size * sizeof(T)));
	if (buffer)
		CudaHelper::checkError(cudaMemcpy(bufferPointer, buffer, size * sizeof(T), cudaMemcpyHostToDevice));
}