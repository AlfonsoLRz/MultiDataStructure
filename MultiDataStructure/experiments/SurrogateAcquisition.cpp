#include "../stdafx.h"
#include "SurrogateAcquisition.h"

namespace Experiments
{
	SurrogateAcquisition loadSurrogateAcquisition(const std::string& modelPath)
	{
		SurrogateAcquisition acquisition;
		acquisition.modelPath = modelPath;
		if (modelPath.empty())
			return acquisition;

		try
		{
			acquisition.model = loadSchemaSelectorModel(modelPath);
		}
		catch (const std::exception& exception)
		{
			std::cerr << "  surrogate acquisition: model load failed (" << exception.what() << "); falling back to mutation-only GA\n";
			return acquisition;
		}

		// Measured-best artifacts cannot rank generic generated genomes — they only know the
		// single tuned schema. ONNX rankers are supported in principle but routing them through
		// `scoreSchemaCandidates` requires the ONNX runtime to be linked; we degrade gracefully
		// rather than throw inside the GA's hot path.
		if (acquisition.model.measuredBestSelector)
		{
			std::cerr << "  surrogate acquisition: '" << modelPath
				<< "' is a measured_best_schema artifact; it cannot rank arbitrary generated genomes. Provide a linear or ONNX score ranker.\n";
			return acquisition;
		}

		acquisition.active = true;
		return acquisition;
	}

	std::vector<SchemaCandidate> acquireSurrogateProposals(
		const SurrogateAcquisition& acquisition,
		const SchemaGenerationOptions& generationOptions,
		size_t poolSize,
		size_t topK,
		const PointCloud& cloud,
		const WorkloadProfile& workload,
		uint32_t seed)
	{
		if (!acquisition.active || poolSize == 0 || topK == 0)
			return {};

		SchemaGenerationOptions poolOptions = generationOptions;
		poolOptions.count = poolSize;
		poolOptions.seed = seed;
		std::vector<SchemaCandidate> pool = generateSchemaCandidates(poolOptions);
		if (pool.empty())
			return {};

		std::vector<CandidatePrediction> predictions;
		try
		{
			predictions = scoreSchemaCandidates(acquisition.model, workload, cloud, pool);
		}
		catch (const std::exception& exception)
		{
			std::cerr << "  surrogate acquisition: scoring failed (" << exception.what() << ")\n";
			return {};
		}

		std::vector<size_t> order(pool.size());
		std::iota(order.begin(), order.end(), size_t(0));
		std::sort(order.begin(), order.end(), [&predictions](size_t a, size_t b) {
			return predictions[a].predictedScore < predictions[b].predictedScore;
		});

		const size_t selectCount = std::min(topK, pool.size());
		std::vector<SchemaCandidate> proposals;
		proposals.reserve(selectCount);
		for (size_t i = 0; i < selectCount; ++i)
			proposals.push_back(std::move(pool[order[i]]));
		return proposals;
	}
}
