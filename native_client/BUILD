# Description: Deepspeech native client library.

cc_binary(
    name = "libdeepspeech.so",
    srcs = ["deepspeech.cc", "deepspeech.h"],
    deps = [":c_speech_features",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:direct_session",
            "//tensorflow/core/kernels:logging_ops",

            # The following is core:all_kernels, minus the parts not necessary
            # for DeepSpeech

            "//tensorflow/core/kernels:array",
#            "//tensorflow/core/kernels:candidate_sampler_ops",
            "//tensorflow/core/kernels:control_flow_ops",
            "//tensorflow/core/kernels:ctc_ops",
            "//tensorflow/core/kernels:data_flow",
#            "//tensorflow/core/kernels:fake_quant_ops",
#            "//tensorflow/core/kernels:function_ops",
#            "//tensorflow/core/kernels:image",
#            "//tensorflow/core/kernels:io",
#            "//tensorflow/core/kernels:linalg",
#            "//tensorflow/core/kernels:logging",
            "//tensorflow/core/kernels:math",
#            "//tensorflow/core/kernels:multinomial_op",
            "//tensorflow/core/kernels:nn",
#            "//tensorflow/core/kernels:parameterized_truncated_normal_op",
#            "//tensorflow/core/kernels:parsing",
#            "//tensorflow/core/kernels:random_ops",
#            "//tensorflow/core/kernels:required",
#            "//tensorflow/core/kernels:resource_variable_ops",
#            "//tensorflow/core/kernels:sdca_ops",
#            "//tensorflow/core/kernels:set_kernels",
            "//tensorflow/core/kernels:sparse",
#            "//tensorflow/core/kernels:state",
#            "//tensorflow/core/kernels:string",
#            "//tensorflow/core/kernels:training_ops",
#            "//tensorflow/core/kernels:word2vec_kernels",
    ],
    linkshared = 1,
)

# Note, we need gnu99 as opposed to c99 as the latter doesn't define M_PI
# Also, fma/avx/avx2 enable in gcc cause significant performance decreases in
# c_speech_features and so are force-disabled.
cc_library(
    name="c_speech_features",
    srcs = ["c_speech_features/c_speech_features.c",
            "kiss_fft130/kiss_fft.c",
            "kiss_fft130/tools/kiss_fftr.c"],
    hdrs = ["c_speech_features/c_speech_features.h",
            "c_speech_features/c_speech_features_config.h",
            "kiss_fft130/kiss_fft.h",
            "kiss_fft130/_kiss_fft_guts.h",
            "kiss_fft130/tools/kiss_fftr.h"],
    includes = ["c_speech_features",
                "kiss_fft130"],
    copts = ["-std=gnu99", "-mno-fma", "-mno-avx", "-mno-avx2"],
    nocopts = "(-fstack-protector|-fno-omit-frame-pointer)",
    linkstatic = 1,
)
