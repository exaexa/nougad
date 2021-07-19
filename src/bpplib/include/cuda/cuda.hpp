/*
* Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
* Last Modification: 24.12.2015
* License: CC 3.0 BY-NC (http://creativecommons.org/)
*/
#ifndef BPPLIB_CUDA_CUDA_HPP
#define BPPLIB_CUDA_CUDA_HPP

#include <misc/exception.hpp>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>


namespace bpp
{

/**
 * \brief A stream exception that is base for all runtime errors.
 */
class CudaError : public RuntimeError
{
protected:
	cudaError_t mStatus;

public:
	CudaError(cudaError_t status = cudaSuccess) : RuntimeError(), mStatus(status) {}
	CudaError(const char *msg, cudaError_t status = cudaSuccess) : RuntimeError(msg), mStatus(status) {}
	CudaError(const std::string &msg, cudaError_t status = cudaSuccess) : RuntimeError(msg), mStatus(status) {}
	virtual ~CudaError() noexcept {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	CudaError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * \brief CUDA error code check.
 *		This is internal function used by CUCH macro.
 */
inline void _cuda_check(cudaError_t status, int line, const char *srcFile, const char *errMsg = nullptr)
{
	if (status != cudaSuccess) {
		throw (bpp::CudaError(status) << "CUDA Error (" << status << "): " << cudaGetErrorString(status) << "\n"
			<< "at " << srcFile << "[" << line << "]: " << errMsg);
	}
}

/**
 * \brief Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) bpp::_cuda_check(status, __LINE__, __FILE__, #status)




/**
 * \brief Buffer of fixed size allocated by CUDA host alloc.
 *		Data must be in this buffer to achieve data transfer overlapping.
 * \tparam T Type of the items in the buffer.
 */
template<typename T>
class HostBuffer
{
private:
	T* mData;				///< Internal pointer to the buffer.
	std::size_t mCount;		///< Number of items (of type T) in the buffer.
	int mDevice;			///< ID of the device for which the buffer was allocated.
	unsigned mFlags;		///< Flags used for allocation (i.e., whether the memory is mapped, pinned, ...).

public:
	/**
	 * \brief Initialize (and allocate) the host buffer.
	 * \param count Number of elements in the buffer.
	 * \param mapped Makes the buffer mapped to the GPU global memory.
	 * \param pinned Marks the memory as pinned (cudaPortable). Pinned memory cannot be swapped out.
	 * \param writeCombined Marks the memory write-combined (not cached by CPU).
	 *		WC memory is faster for host -> GPU transfers, but slower when readed by CPU.
	 * \param deviceId Id of a device for which the buffer is primarily designed.
	 *		The buffer may be used with any device, but only selected device may benefit from
	 *		memory transfer overlapping.
	 */
	HostBuffer<T>(std::size_t count = 0, bool mapped = false, bool pinned = false, bool writeCombined = false, int deviceId = -1)
		: mData(nullptr), mCount(0), mDevice(-1), mFlags(0)
	{
		realloc(count, mapped, pinned, writeCombined, deviceId);
	}


	// Destructor releases the internal memory.
	virtual ~HostBuffer<T>()
	{
		try {
			free();
		}
		catch (std::exception&) {}
	}


	// Move constructor.
	HostBuffer<T>(HostBuffer<T> &&buf)
	{
		mData = buf.mData;
		mCount = buf.mCount;
		buf.mData = nullptr;
		buf.mCount = 0;
	}


	// Copy constructor is disabled, since we do not have reference counting.
	HostBuffer<T>(const HostBuffer<T> &buf) = delete;


	/**
	 * \brief Get number of items in the buffer.
	 */
	std::size_t size() const
	{
		return mCount;
	}

	/**
	 * \brief Constant accessor to the allocated pointer.
	 */
	const T& operator*() const
	{
		return *mData;
	}


	/**
	 * \brief Writeable accessor to the allocated pointer.
	 */
	T& operator*()
	{
		return *mData;
	}


	/**
	 * \brief Unchecked constant accessor.
	 */
	const T& operator[](std::size_t idx) const
	{
		return mData[idx];
	}


	/**
	 * \brief Unchecked writeable accessor.
	 */
	T& operator[](std::size_t idx)
	{
		return mData[idx];
	}

	/**
	 * \brief Checked constant accessor.
	 */
	const T& at(std::size_t idx) const
	{
		if (idx >= mCount)
			throw (bpp::RuntimeError() << "Index (" << idx << ") is out of range (" << mCount << ").");
		return mData[idx];
	}


	/**
	 * \brief Checked writeable acessor.
	 */
	T& at(std::size_t idx)
	{
		if (idx >= mCount)
			throw (bpp::RuntimeError() << "Index (" << idx << ") is out of range (" << mCount << ").");
		return mData[idx];
	}


	/**
	 * \brief Release the buffer.
	 */
	void free()
	{
		if (mData != nullptr) {
			CUCH(cudaFreeHost(mData));
			mData = nullptr;
			mCount = 0;
			mFlags = 0;
			mDevice = -1;
		}
	}


	/**
	 * \brief Dispose the current buffer and allocate a new one.
	 * \param count Number of elements in the buffer.
	 * \param mapped Makes the buffer mapped to the GPU global memory.
	 * \param pinned Marks the memory as pinned (cudaPortable). Pinned memory cannot be swapped out.
	 * \param writeCombined Marks the memory write-combined (not cached by CPU).
	 *		WC memory is faster for host -> GPU transfers, but slower when readed by CPU.
	 * \param deviceId Id of a device for which the buffer is primarily designed.
	 *		The buffer may be used with any device, but only selected device may benefit from
	 *		memory transfer overlapping.
	 */
	void realloc(std::size_t count, bool mapped = false, bool pinned = false, bool writeCombined = false, int deviceId = -1)
	{
		// Release previously allocated memory...
		free();
		if (count == 0)
			return;

		// Set the proper device.
		if (deviceId >= 0)
			CUCH(cudaSetDevice(deviceId));

		// Allocate the memory.
		mFlags = (mapped ? cudaHostAllocMapped : 0)
			|| (pinned ? cudaHostAllocPortable : 0)
			|| (writeCombined ? cudaHostAllocWriteCombined : 0);
		CUCH(cudaHostAlloc(&mData, count * sizeof(T), mFlags));
		mCount = count;
		mDevice = deviceId;
	}


	/**
	 * \brief In case of mapped buffers, retrieves the GPU pointer that can be used in a kernel.
	 */
	T* getDevicePtr()
	{
		if (mFlags & cudaHostAllocMapped) {
			T* res;
			CUCH(cudaHostGetDevicePointer(&res, mData, 0));
			return res;
		}
		else
			return nullptr;
	}
};





/**
 * \brief Buffer of fixed size allocated for a CUDA device.
 * \tparam T Type of the items in the buffer.
 */
template<typename T>
class CudaBuffer
{
private:
	T* mData;
	std::size_t mCount;


	/**
	 * \brief Check (and fix) count and offset parameters of a read operation.
	 * \param count Number of elements being read. The value is saturated if exceeds maximum.
	 * \param offset The index of the first element being read.
	 * \return True if the operation should continue, false if it shoud terminate silently.
	 */
	bool rangeCheck(std::size_t &count, std::size_t offset)
	{
		if (!mData)
			throw (bpp::RuntimeError() << "Unable to read not allocated buffer.");

		if (offset >= mCount)
			throw (bpp::RuntimeError() << "Given offset " << offset << " is beyond the size of the buffer (" << mCount << ").");
		count = std::min(count, mCount - offset);
		return (count > 0);
	}

public:
	CudaBuffer<T>(std::size_t count = 0)
		: mData(nullptr), mCount(0)
	{
		realloc(count);
	}

	/**
	 * \brief Raw buffer accessor.
	 * \return Pointer that can be used in kernel calls.
	 */
	T* operator*()
	{
		return mData;
	}


	/**
	 * \brief Constant raw buffer accessor.
	 * \return Pointer that can be used in kernel calls.
	 */
	const T* operator*() const
	{
		return mData;
	}


	/**
	 * \brief Return the size of the buffer (in multiples of T).
	 */
	std::size_t size() const
	{
		return mCount;
	}

	/**
	 * \bried Dispose of the buffer. Same as realloc the buffer to size 0.
	 */
	void free()
	{
		if (mCount > 0) {
			CUCH(cudaFree(mData));
			mData = nullptr;
			mCount = 0;
		}
	}


	/**
	 * \bried Change buffer size. All data are lost.
	 * \param count New buffer size in multiples of T.
	 */
	void realloc(std::size_t count)
	{
		if (count == mCount) return;

		free();
		if (count == 0)
			return;

		CUCH(cudaMalloc(&mData, count * sizeof(T)));
		mCount = count;
	}


	/**
	 * \brief Fill entire buffer with one value.
	 * \param value The value used for filling (must fit a byte/char).
	 */
	void memset(int value)
	{
		if (!mData)
			throw (bpp::RuntimeError() << "Unable to set memroy of an empty buffer.");
		CUCH(cudaMemset(mData, value, mCount * sizeof(T)));
	}


	/**
	 * \brief Fill entire buffer with one value asynchronously.
	 * \param value The value used for filling (must fit a byte/char).
	 * \param stream CUDA stream used to handle the async operation.
	 */
	void memsetAsync(int value, cudaStream_t stream)
	{
		if (!mData)
			throw (bpp::RuntimeError() << "Unable to set memroy of an empty buffer.");
		CUCH(cudaMemsetAsync(mData, value, mCount * sizeof(T), stream));
	}



	/*
	 * Read Operations (GPU -> Host transfers)
	 */

	/**
	 * \brief Read the CUDA buffer and store its data to regular (host memory) buffer.
	 * \param buf Pointer to host memory, where the data will be stored.
	 * \param count Number of items being read (if not specified, entire buffer is read).
	 * \param offset Offset in the CUDA buffer from which the reading is performed.
	 */
	void read(T* buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (!rangeCheck(count, offset)) return;
		CUCH(cudaMemcpy(buf, mData + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
	}


	/**
	 * \brief Read the CUDA buffer and store its data a vector in host memory.
	 * \param buf Reference to a STL vector, where the data will be stored. Vector is grown if necessary.
	 * \param count Number of items being read (if not specified, entire buffer is read).
	 * \param offset Offset in the CUDA buffer from which the reading is performed.
	 */
	void read(std::vector<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (!rangeCheck(count, offset)) return;

		if (buf.size() < count)
			buf.resize(count);

		CUCH(cudaMemcpy(&buf[0], mData + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
	}


	/**
	 * \brief Read the CUDA buffer and store its data to a HostBuffer.
	 * \param buf Reference to a HostBuffer object where the data will be copied.
	 * \param count Number of items being read (if not specified, entire buffer is read).
	 * \param offset Offset in the CUDA buffer from which the reading is performed.
	 */
	void read(HostBuffer<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (buf.size() == 0)
			throw (bpp::RuntimeError() << "Unable to copy data into not allocated HostBuffer.");
		count = std::min(count, buf.size());
		read(&(*buf), count, offset);
	}


	/**
	* \brief Read the CUDA buffer and store its data to regular (host memory) buffer.
	* \param buf Pointer to host memory, where the data will be stored.
	* \param count Number of items being read (if not specified, entire buffer is read).
	* \param offset Offset in the CUDA buffer from which the reading is performed.
	*/
	void readAsync(cudaStream_t stream, T* buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (!rangeCheck(count, offset)) return;
		CUCH(cudaMemcpyAsync(buf, mData + offset, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}


	/**
	 * \brief Read the CUDA buffer and store its data a vector in host memory.
	 * \param buf Reference to a STL vector, where the data will be stored. Vector is grown if necessary.
	 * \param count Number of items being read (if not specified, entire buffer is read).
	 * \param offset Offset in the CUDA buffer from which the reading is performed.
	 */
	void readAsync(cudaStream_t stream, std::vector<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (!rangeCheck(count, offset)) return;

		if (buf.size() < count)
			buf.resize(count);

		CUCH(cudaMemcpyAsync(&buf[0], mData + offset, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}

	/**
	 * \brief Read the CUDA buffer and store its data to a HostBuffer.
	 * \param buf Reference to a HostBuffer object where the data will be copied.
	 * \param count Number of items being read (if not specified, entire buffer is read).
	 * \param offset Offset in the CUDA buffer from which the reading is performed.
	 */
	void readAsync(cudaStream_t stream, HostBuffer<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (buf.size() == 0)
			throw (bpp::RuntimeError() << "Unable to copy data into not allocated HostBuffer.");
		count = std::min(count, buf.size());
		readAsync(stream, &(*buf), count, offset);
	}



	/*
	 * Write Operations (Host -> GPU transfers)
	 */

	/**
	 * \brief Write data from host memory to the CUDA buffer.
	 * \param buf Pointer to host memory, from which the data are copied.
	 * \param count Number of items being written (if not specified, entire buffer is filled).
	 * \param offset Offset in the CUDA buffer from which the writting is performed.
	 */
	void write(const T* buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (!rangeCheck(count, offset)) return;
		CUCH(cudaMemcpy(mData + offset, buf, count * sizeof(T), cudaMemcpyHostToDevice));

	}


	/**
	 * \brief Write data from a vector to the CUDA buffer.
	 * \param buf Reference to a STL vector, from which the data will be copied.
	 * \param count Number of items being written (if not specified, entire buffer is filled).
	 * \param offset Offset in the CUDA buffer from which the writting is performed.
	 */
	void write(const std::vector<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		rangeCheck(count, offset);
		count = std::min(count, buf.size());
		if (count == 0) return;

		CUCH(cudaMemcpy(mData + offset, &buf[0], count * sizeof(T), cudaMemcpyHostToDevice));
	}


	/**
	 * \brief Copy data from a HostBuffer into CUDA buffer.
	 * \param buf Reference to a HostBuffer object from which the data will be copied.
	 * \param count Number of items being written (if not specified, entire buffer is filled).
	 * \param offset Offset in the CUDA buffer from which the writting is performed.
	 */
	void write(const HostBuffer<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (buf.size() == 0) {
			throw (bpp::RuntimeError() << "Unable to copy data from not allocated HostBuffer.");
		}
		count = std::min(count, buf.size());
		write(&(*buf), count, offset);
	}


	/**
	 * \brief Write data from host memory to the CUDA buffer.
	 * \param buf Pointer to host memory, from which the data are copied.
	 * \param count Number of items being written (if not specified, entire buffer is filled).
	 * \param offset Offset in the CUDA buffer from which the writting is performed.
	 */
	void writeAsync(cudaStream_t stream, const T* buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (!rangeCheck(count, offset)) return;
		CUCH(cudaMemcpy(mData + offset, buf, count * sizeof(T), cudaMemcpyHostToDevice));

	}


	/**
	 * \brief Write data from a vector to the CUDA buffer.
	 * \param buf Reference to a STL vector, from which the data will be copied.
	 * \param count Number of items being written (if not specified, entire buffer is filled).
	 * \param offset Offset in the CUDA buffer from which the writting is performed.
	 */
	void writeAsync(cudaStream_t stream, const std::vector<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		rangeCheck(count, offset);
		count = std::min(count, buf.size());
		if (count == 0) return;

		CUCH(cudaMemcpy(mData + offset, &buf[0], count * sizeof(T), cudaMemcpyHostToDevice));
	}


	/**
	 * \brief Copy data from a HostBuffer into CUDA buffer.
	 * \param buf Reference to a HostBuffer object from which the data will be copied.
	 * \param count Number of items being written (if not specified, entire buffer is filled).
	 * \param offset Offset in the CUDA buffer from which the writting is performed.
	 */
	void writeAsync(cudaStream_t stream, const HostBuffer<T> &buf, std::size_t count = ~(std::size_t)0, std::size_t offset = 0)
	{
		if (buf.size() == 0)
			throw (bpp::RuntimeError() << "Unable to copy data from not allocated HostBuffer.");
		count = std::min(count, buf.size());
		writeAsync(stream, &(*buf), count, offset);
	}
};



/**
 * \brief Cuda device wrapper
 */
class CudaDevice
{
public:
	/**
	 * \brief Return total number of available devices.
	 */
	static std::size_t count()
	{
		int count;
		CUCH(cudaGetDeviceCount(&count));
		return (std::size_t)count;
	}

	/**
	 * \brief Set given device as current device.
	 */
	static void select(std::size_t device)
	{
		CUCH(cudaSetDevice((int)device));
	}
};



/**
 * \brief Wrapper for CUDA stream.
 */
class CudaStream
{
private:
	std::size_t mDevice;	///< Device for which the 
	int mFlags;				///< Stream flags (used for creation)
	cudaStream_t mStream;	///< The underlying CUDA stream.

public:
	CudaStream(std::size_t device, bool nonBlocking = false) : mDevice(device)
	{
		mFlags = nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
		CudaDevice::select(device);
		CUCH(cudaStreamCreateWithFlags(&mStream, mFlags));
	}

	~CudaStream()
	{
		cudaStreamDestroy(mStream);		// ignoring the return code
	}

	cudaStream_t operator*() const
	{
		return mStream;
	}

	std::size_t getDeviceId() const
	{
		return mDevice;
	}

	void selectDevice() const
	{
		CudaDevice::select(mDevice);
	}

	bool finished()
	{
		cudaError_t res = cudaStreamQuery(mStream);
		if (res == cudaErrorNotReady) return false;
		CUCH(res);
		return true;
	}


	void synchronize()
	{
		CUCH(cudaStreamSynchronize(mStream));
	}
};



}
#endif
