// Kino Bloom : https://github.com/keijiro/KinoBloom

using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class ComputeBloomRendererFeature : ScriptableRendererFeature
{
    class ComputeBloomPass : ScriptableRenderPass
    {
        private ComputeShader _bloomCS;
        
        private int _prefilterKernelHandle;
        private int _downSampleKernelHandle;
        private int _upSampleKernelHandle;
        private int _combinerKernelHandle;

        private RenderTextureDescriptor _cameraTargetDescriptor;
        private int _cameraTargetPixelWidth;
        private int _cameraTargetPixelHeight;

        private string _downTextureName = "_bloomDown";
        private string _upTextureName = "_bloomUp";
        private string _finalTextureName = "_finalTex";
        
        const int _maxIterations = 16;

        private bool _highQuality;
        private float _radius;
        private float _thresholdLinear;
        private float _softKnee;
        private float _intensity;
        public ComputeBloomPass(RenderPassEvent passEvent, bool highQuality, float radius, float thresholdLinear, float softKnee, float intensity)
        {
            renderPassEvent = passEvent;
            
            _bloomCS = Resources.Load<ComputeShader>("ComputeShaderBloom");
            
            _prefilterKernelHandle = _bloomCS.FindKernel("CSPrefilter");
            _downSampleKernelHandle = _bloomCS.FindKernel("CSDownsampling");
            _upSampleKernelHandle = _bloomCS.FindKernel("CSUpsampling");
            _combinerKernelHandle = _bloomCS.FindKernel("CSCombiner");
            
            _highQuality = highQuality;
            _radius = radius;
            _thresholdLinear = thresholdLinear;
            _softKnee = softKnee;
            _intensity = intensity;
        }
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            CameraData cameraData = renderingData.cameraData;
            _cameraTargetDescriptor = cameraData.cameraTargetDescriptor;
            _cameraTargetPixelWidth = _cameraTargetDescriptor.width;
            _cameraTargetPixelHeight = _cameraTargetDescriptor.height;

            _bloomCS.SetInts("_InputWidth", _cameraTargetPixelWidth);
            _bloomCS.SetInts("_InputHeight", _cameraTargetPixelHeight);
            
            if (_highQuality)
            {
                cmd.EnableKeyword(_bloomCS, new LocalKeyword(_bloomCS, "HIGH_QUALITY"));
            }
            else
            {
                cmd.DisableKeyword(_bloomCS, new LocalKeyword(_bloomCS, "HIGH_QUALITY"));
            }
        }

        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CameraData cameraData = renderingData.cameraData;
            CommandBuffer cmd = CommandBufferPool.Get();

            cmd.BeginSample("ComputeBloom");

            RunBloom(cameraData, cmd, ref renderingData);

            cmd.EndSample("ComputeBloom");
            context.ExecuteCommandBuffer(cmd);
            cmd.Release();
        }

        private RenderTextureDescriptor GetCompatibleDescriptor(RenderTextureDescriptor origin, int width, int height)
        {
            RenderTextureDescriptor desc = origin;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;
            desc.width = width;
            desc.height = height;
            desc.graphicsFormat = GraphicsFormat.B10G11R11_UFloatPack32;
            desc.enableRandomWrite = true;
            return desc;
        }

        private void RunBloom(CameraData cameraData, CommandBuffer cmd, ref RenderingData renderingData)
        {
            //0. Variable Set-up
            var logh = Mathf.Log(_cameraTargetPixelHeight, 2) + _radius - 8;
            var logh_i = (int)logh;
            var iterations = Mathf.Clamp(logh_i, 1, _maxIterations);
            
            var lthresh = _thresholdLinear;

            var knee = lthresh * _softKnee + 1e-5f;
            var curve = new Vector3(lthresh - knee, knee * 2, 0.25f / knee);

            cmd.SetComputeFloatParam(_bloomCS, "_Threshold", lthresh);
            cmd.SetComputeFloatParam(_bloomCS, "_Intensity", _intensity);
            cmd.SetComputeFloatParam(_bloomCS, "_SampleScale", 0.5f + logh - logh_i);
            cmd.SetComputeVectorParam(_bloomCS, "_Curve", curve);

            //1. Prefilter
            var halfSizeRTDesc = GetCompatibleDescriptor(_cameraTargetDescriptor, _cameraTargetPixelWidth / 2 , _cameraTargetPixelHeight / 2);
            cmd.GetTemporaryRT(Shader.PropertyToID(_downTextureName + "0"), halfSizeRTDesc, FilterMode.Bilinear);
            RunCompute(cmd, _prefilterKernelHandle, BuiltinRenderTextureType.CurrentActive, Shader.PropertyToID(_downTextureName + "0"), halfSizeRTDesc.width, halfSizeRTDesc.height);

            //2. DownSampling
            RTBundle lastTexBundle = new RTBundle(Shader.PropertyToID(_downTextureName + "0"), halfSizeRTDesc);
            
            List<RTBundle> downBufferList = new List<RTBundle> { new RTBundle(Shader.PropertyToID(_downTextureName + "0"), halfSizeRTDesc) };
            for (var level = 1; level < iterations; level++)
            {
                downBufferList.Add(new RTBundle(Shader.PropertyToID(_downTextureName + level), 
                    GetCompatibleDescriptor(_cameraTargetDescriptor, lastTexBundle.Desc.width / 2, lastTexBundle.Desc.height / 2)));
                
                cmd.GetTemporaryRT(Shader.PropertyToID(_downTextureName + level), downBufferList[level].Desc, FilterMode.Bilinear);
                RunCompute(cmd, _downSampleKernelHandle, lastTexBundle, downBufferList[level]);

                lastTexBundle = downBufferList.Last();
            }
            
            //3. UpSampling
            List<RTBundle> upBufferList = new List<RTBundle>();
            for (var level = iterations - 2; level >= 0; level--)
            {
                var baseTexDesc = downBufferList[level].Desc;
                cmd.SetComputeTextureParam(_bloomCS, _upSampleKernelHandle, "UpSampleBase",downBufferList[level].ID);
                
                upBufferList.Add(new RTBundle(Shader.PropertyToID(_upTextureName + level),
                    GetCompatibleDescriptor(_cameraTargetDescriptor, baseTexDesc.width, baseTexDesc.height)));
                
                cmd.GetTemporaryRT(Shader.PropertyToID(_upTextureName + level), upBufferList.Last().Desc, FilterMode.Bilinear);
                RunCompute(cmd, _upSampleKernelHandle, lastTexBundle, upBufferList.Last());

                lastTexBundle = upBufferList.Last();
            }
            
            //4. Final (combine)
            RTBundle finalTexBundle = new RTBundle(Shader.PropertyToID(_finalTextureName), GetCompatibleDescriptor(_cameraTargetDescriptor, _cameraTargetPixelWidth, _cameraTargetPixelHeight));
            cmd.GetTemporaryRT(Shader.PropertyToID(_finalTextureName), finalTexBundle.Desc, FilterMode.Bilinear);
            
            cmd.SetComputeTextureParam(_bloomCS, _combinerKernelHandle, "UpSampleBase", BuiltinRenderTextureType.CurrentActive);
            RunCompute(cmd, _combinerKernelHandle, lastTexBundle, finalTexBundle);
            cmd.Blit(Shader.PropertyToID(_finalTextureName), renderingData.cameraData.renderer.cameraColorTarget);
        }

        private void RunCompute(CommandBuffer cmd, int kernelIndex, RenderTargetIdentifier input, RenderTargetIdentifier output, int outputWidth, int outputHeight)
        {
            cmd.SetComputeTextureParam(_bloomCS, kernelIndex, "Input", input);
            cmd.SetComputeTextureParam(_bloomCS, kernelIndex, "Output", output);
            _bloomCS.GetKernelThreadGroupSizes(kernelIndex, out uint kernelX, out uint kernelY, out uint _);
            cmd.DispatchCompute(_bloomCS, kernelIndex, Mathf.Max(outputWidth / (int)kernelX, 1), Mathf.Max(outputHeight / (int)kernelY, 1), 1);
        }
        
        private void RunCompute(CommandBuffer cmd, int kernelIndex, RTBundle input, RTBundle output)
        {
            cmd.SetComputeIntParam(_bloomCS, "_InputWidth", input.Desc.width);
            cmd.SetComputeIntParam(_bloomCS, "_InputHeight", input.Desc.height);
            RunCompute(cmd, kernelIndex, input.ID, output.ID, output.Desc.width, output.Desc.height); 
        }

        private struct RTBundle
        {
            public RenderTargetIdentifier ID;
            public RenderTextureDescriptor Desc;

            public RTBundle(RenderTargetIdentifier id, RenderTextureDescriptor desc)
            {
                ID = id;
                Desc = desc;
            }
        }
        
    }
    
    ComputeBloomPass _computeBloomPass;

    public override void Create()
    {
        _computeBloomPass = new ComputeBloomPass(RenderPassEvent.AfterRenderingPostProcessing, highQuality, radius, thresholdLinear, softKnee, intensity);
    }
    
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(_computeBloomPass);
    }
    
#region Public Properties

    /// Prefilter threshold (gamma-encoded)
    /// Filters out pixels under this level of brightness.
    public float thresholdGamma {
        get { return Mathf.Max(_threshold, 0); }
        set { _threshold = value; }
    }

    /// Prefilter threshold (linearly-encoded)
    /// Filters out pixels under this level of brightness.
    public float thresholdLinear {
        get { return GammaToLinear(thresholdGamma); }
        set { _threshold = LinearToGamma(value); }
    }

    [SerializeField]
    [Tooltip("Filters out pixels under this level of brightness.")]
    float _threshold = 0.8f;

    /// Soft-knee coefficient
    /// Makes transition between under/over-threshold gradual.
    public float softKnee {
        get { return _softKnee; }
        set { _softKnee = value; }
    }

    [SerializeField, Range(0, 1)]
    [Tooltip("Makes transition between under/over-threshold gradual.")]
    float _softKnee = 0.5f;

    /// Bloom radius
    /// Changes extent of veiling effects in a screen
    /// resolution-independent fashion.
    public float radius {
        get { return _radius; }
        set { _radius = value; }
    }

    [SerializeField, Range(1, 7)]
    [Tooltip("Changes extent of veiling effects\n" +
             "in a screen resolution-independent fashion.")]
    float _radius = 2.5f;

    /// Bloom intensity
    /// Blend factor of the result image.
    public float intensity {
        get { return Mathf.Max(_intensity, 0); }
        set { _intensity = value; }
    }

    [SerializeField]
    [Tooltip("Blend factor of the result image.")]
    float _intensity = 0.8f;

    /// High quality mode
    /// Controls filter quality and buffer resolution.
    public bool highQuality {
        get { return _highQuality; }
        set { _highQuality = value; }
    }

    [SerializeField]
    [Tooltip("Controls filter quality and buffer resolution.")]
    bool _highQuality = true;
    
#endregion

#region MonoBehaviour Functions

    float LinearToGamma(float x)
    {
#if UNITY_5_3_OR_NEWER
        return Mathf.LinearToGammaSpace(x);
#else
        if (x <= 0.0031308f)
            return 12.92f * x;
        else
            return 1.055f * Mathf.Pow(x, 1 / 2.4f) - 0.055f;
#endif
    }

    float GammaToLinear(float x)
    {
#if UNITY_5_3_OR_NEWER
        return Mathf.GammaToLinearSpace(x);
#else
        if (x <= 0.04045f)
            return x / 12.92f;
        else
            return Mathf.Pow((x + 0.055f) / 1.055f, 2.4f);
#endif
    }
#endregion
}


