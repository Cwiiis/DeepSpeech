#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "deepspeech.h"

using namespace tensorflow;

struct _DeepSpeechContext {
  Session* session;
  GraphDef graph_def;
};

DeepSpeechContext*
DsInit(const char* aModelPath)
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

  /*// Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 3.0;

  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2.0;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
    { "b", b },
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  status = session->Run(inputs, {"c"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;*/

