
#ifndef __DEEPSPEECH_H__
#define __DEEPSPEECH_H__

typedef struct _DeepSpeechContext DeepSpeechContext;

DeepSpeechContext* DsInit(const char* aModelPath);
void DsClose(DeepSpeechContext* aCtx);

#endif /* __DEEPSPEECH_H__ */
