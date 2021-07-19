#pragma once

#include <algorithm>

#include "interface.hpp"

/**
 * Serial implementation of GradientDescend algorithm.
 */
template <typename F = float>
class GradientDescendSerialAlgorithm : public IGradientDescendAlgorithm<F>
{
private:
	const DataPoints<F>*mPoints, *mSpectra, *mSpectraPositiveWeights, *mSpectraNegativeWeights, *mResultWeights;
	std::vector<F> mGradientMemory;

public:
	GradientDescendSerialAlgorithm()
		: mPoints(nullptr), mSpectra(nullptr), mSpectraPositiveWeights(nullptr), mSpectraNegativeWeights(nullptr), mResultWeights(nullptr)
	{}

	virtual void initialize(const DataPoints<F>& points, const DataPoints<F>& spectra, const DataPoints<F>& spectraPositiveWeights,
							const DataPoints<F>& spectraNegativeWeights, const DataPoints<F>& resultWeights, DataPoints<F>& initialResults,
							const std::size_t iterations, const F alpha, const F acceleration)
	{
		IGradientDescendAlgorithm::initialize(points, spectra, spectraPositiveWeights, spectraNegativeWeights, resultWeights, initialResults,
											  iterations, alpha, acceleration);
		mPoints = &points;
		mSpectra = &spectra;
		mSpectraPositiveWeights = &spectraPositiveWeights;
		mSpectraNegativeWeights = &spectraNegativeWeights;
		mResultWeights = &resultWeights;
		this->mResult = std::move(initialResults);
	}

	void prepareInputs() override { mGradientMemory.resize(this->mSpectrumN, F(0)); }

	void run() override
	{
		for (size_t n_idx = 0; n_idx < this->mN; ++n_idx)
		{
			F* __restrict result = this->mResult[n_idx];
			F* __restrict resultResidual = this->mResultResiduals[n_idx];
			const F* __restrict point = (*mPoints)[n_idx];

			memset(mGradientMemory.data(), 0, sizeof(F) * this->mSpectrumN);

			for (size_t i = 0;; ++i)
			{
				/* compute the residuals */
				{
					for (size_t d_idx = 0; d_idx < this->mDim; ++d_idx)
						resultResidual[d_idx] = -point[d_idx];

					for (size_t k_idx = 0; k_idx < this->mSpectrumN; ++k_idx)
						for (size_t d_idx = 0; d_idx < this->mDim; ++d_idx)
							resultResidual[d_idx] += result[k_idx] * (*mSpectra)[k_idx][d_idx];
				}

				if (i >= this->mIterations)
					break;

				for (size_t k_idx = 0; k_idx < this->mSpectrumN; ++k_idx)
				{
					/* guess the direction */
					F gradient = result[k_idx] > 0 ? 0 : (*mResultWeights)[0][k_idx] * result[k_idx];
					for (size_t d_idx = 0; d_idx < this->mDim; ++d_idx)
					{
						gradient +=
							resultResidual[d_idx] * (*mSpectra)[k_idx][d_idx]
							* (resultResidual[d_idx] > 0 ? (*mSpectraPositiveWeights)[k_idx][d_idx] : (*mSpectraNegativeWeights)[k_idx][d_idx]);
					}

					/* apply the gradient */
					gradient *= this->mAlpha;
					if (gradient * mGradientMemory[k_idx] > 0)
						gradient += this->mAcceleration * mGradientMemory[k_idx];
					result[k_idx] -= gradient;
					mGradientMemory[k_idx] = gradient;
				}
			}
		}
	}
};
