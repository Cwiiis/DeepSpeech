
#ifndef __DEEPSPEECH_H__
#define __DEEPSPEECH_H__

typedef struct _DeepSpeechContext DeepSpeechContext;

DeepSpeechContext* DsInit(const char* aModelPath, int aNCep, int aNContext);
void DsClose(DeepSpeechContext* aCtx);

char* DsInfer(DeepSpeechContext* aCtx, float** aBuffer, int aNFrames);

#endif /* __DEEPSPEECH_H__ */
