#pragma once

#include "stdafx.h"

class ApplicationState
{
public:
	// Application
	glm::vec3						_backgroundColor;
	bool							_enableSky;
	float							_materialScattering;
	uint16_t						_numFps;
	glm::uint						_renderBvhNodes;
	uint8_t							_selectedCamera;
	glm::ivec2						_viewportSize;

	// Screenshot
	char							_screenshotFilenameBuffer[60];
	float							_screenshotFactor;
	bool							_transparentScreenshot;

	// Lighting
	float							_gamma, _exposure;
	float							_addition, _multiplication;
	unsigned						_numBounces;
	unsigned 						_numSamples;

	// Camera
	float							_blurStrength;
	float							_defocusAngle;

	ApplicationState()
	{
		_backgroundColor = glm::vec3(.6f);
		_enableSky = true;
		_materialScattering = 1.0f;
		_numFps = 0;
		_renderBvhNodes = 0;
		_selectedCamera = 0;
		_viewportSize = glm::vec3(0);

		strcpy_s(_screenshotFilenameBuffer, 60, "ScreenshotRGBA.png");
		_screenshotFactor = 2.2f;
		_transparentScreenshot = true;

		_gamma = 3.0f;
		_exposure = 1.2f;
		_addition = 0.0f;
		_multiplication = 1.0f;
		_numBounces = 10;
		_numSamples = 1024;

		_blurStrength = 0.0f;
		_defocusAngle = 0.0f;
	}
};