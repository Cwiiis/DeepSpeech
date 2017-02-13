#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "deepspeech.h"

#define INPUT_SIZE (N_CEP + (2 * N_CEP * N_CONTEXT))

using namespace tensorflow;

struct _DeepSpeechContext {
  Session* session;
  GraphDef graph_def;
  int ncep;
  int ncontext;
};

DeepSpeechContext*
DsInit(const char* aModelPath, int aNCep, int aNContext)
{
  if (!aModelPath) {
    return NULL;
  }

  DeepSpeechContext* ctx = new DeepSpeechContext;

  Status status = NewSession(SessionOptions(), &ctx->session);
  if (!status.ok()) {
    delete ctx;
    return NULL;
  }

  status = ReadBinaryProto(Env::Default(), aModelPath, &ctx->graph_def);
  if (!status.ok()) {
    ctx->session->Close();
    delete ctx;
    return NULL;
  }

  status = ctx->session->Create(ctx->graph_def);
  if (!status.ok()) {
    ctx->session->Close();
    delete ctx;
    return NULL;
  }

  ctx->ncep = aNCep;
  ctx->ncontext = aNContext;

  return ctx;
}

void
DsClose(DeepSpeechContext* aCtx)
{
  if (!aCtx) {
    return;
  }

  aCtx->session->Close();
  delete aCtx;
}

char*
DsInfer(DeepSpeechContext* aCtx, float** aBuffer, int aNFrames)
{
  const int frameSize = aCtx->ncep + (2 * aCtx->ncep * aCtx->ncontext);
  Tensor input(DT_FLOAT, TensorShape({1, aNFrames, frameSize}));

  auto input_mapped = input.tensor<float, 3>();
  for (int i = 0; i < aNFrames; i++) {
    for (int j = 0; j < frameSize; j++) {
      input_mapped(0, i, j) = aBuffer[i][j];
    }
  }

  std::vector<Tensor> outputs;
  Status status =
    aCtx->session->Run({{ "input_node", input }},
                       {"output_node"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Error running session: " << status.ToString() << "\n";
    return NULL;
  }

  // Output is an array of shape (1, n_results, result_length).
  // In this case, n_results is also equal to 1.
  auto output_mapped = outputs[0].tensor<int64, 3>();
  int length = output_mapped.dimension(2) + 1;
  char* output = (char*)malloc(sizeof(char) * length);
  for (int i = 0; i < length - 1; i++) {
    int64 character = output_mapped(0, 0, i);
    output[i] = (character ==  0) ? ' ' : (character + 'a' - 1);
  }
  output[length - 1] = '\0';

  return output;
}
