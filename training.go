package mitie

/*
#cgo LDFLAGS: -lmitie

#include <stdlib.h>

#include "mitie.h"
char **newCharArray(int len);
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// NERTrainer is a named entity recognition trainer.
type NERTrainer struct {
	x *C.mitie_ner_trainer
}

// NewNERTrainer constructs a new NER trainer.  It requires a feature
// extractor model (e.g. total_word_feature_extractor.dat)
func NewNERTrainer(filename string) (*NERTrainer, error) {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	x := C.mitie_create_ner_trainer(cs)
	if x == nil {
		return nil, errors.New("unable to create NERTrainer")
	}
	nt := &NERTrainer{x}
	runtime.SetFinalizer(nt, freeNERTrainer)
	return nt, nil
}

func freeNERTrainer(n *NERTrainer) {
	C.mitie_free(unsafe.Pointer(n.x))
}

// AddInstance adds a training instance to an NERTrainer
func (n *NERTrainer) AddInstance(inst *TrainingInstance) error {
	if c := C.mitie_add_ner_training_instance(n.x, inst.x); c != 0 {
		return fmt.Errorf("error adding training instance: %d", c)
	}
	return nil
}

// Train trains the NERTrainer to produce a NamedEntityExtractor which
// can be saved to disk and loaded later.  numThreads is the number of
// threads to use in training while beta controls the trade-off
// between trying to avoid false alarms but also detecting everything.
// Different values of beta have the following interpretations:
// - beta < 1 indicates that you care more about avoiding false alarms than
//   missing detections.  The smaller you make beta the more the trainer will
//   try to avoid false alarms.
// - beta == 1 indicates that you don't have a preference between avoiding
//   false alarms or not missing detections.  That is, you care about these
//   two things equally.
// - beta > 1 indicates that care more about not missing detections than
//   avoiding false alarms.
func (n *NERTrainer) Train(numThreads int, beta float64) (*NamedEntityExtractor, error) {
	if numThreads <= 0 {
		numThreads = 1
	}
	C.mitie_ner_trainer_set_num_threads(n.x, C.ulong(numThreads))
	if beta <= 0 {
		beta = 0
	}
	C.mitie_ner_trainer_set_beta(n.x, C.double(beta))

	x := C.mitie_train_named_entity_extractor(n.x)
	if x == nil {
		return nil, errors.New("error during training")
	}

	ner := &NamedEntityExtractor{x}
	runtime.SetFinalizer(ner, freeNamedEntityExtractor)
	return ner, nil
}

type TrainingInstance struct {
	x      *C.mitie_ner_training_instance
	tokens []string
}

// NewTrainingInstance constructs a new training instance given a string.
func NewTrainingInstance(s string) (*TrainingInstance, error) {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	toks := C.mitie_tokenize(cs)
	if toks == nil {
		return nil, errors.New("unable to tokenize input")
	}
	defer C.mitie_free(unsafe.Pointer(toks))

	tng := C.mitie_create_ner_training_instance(toks)
	if tng == nil {
		return nil, errors.New("unable to create training instance")
	}

	tokens := []string{}
	p := (*[1 << 30]*C.char)(unsafe.Pointer(toks))
	i := 0
	for p[i] != nil {
		tokens = append(tokens, C.GoString(p[i]))
		i++
	}

	ti := &TrainingInstance{x: tng, tokens: tokens}
	runtime.SetFinalizer(ti, freeTrainingInstance)
	return ti, nil
}

// NewTrainingInstance constructs a new training instance.
func NewTrainingInstanceFromToks(toks []string) (*TrainingInstance, error) {
	// create a char** with our tokens
	rawArr := C.newCharArray(C.int(len(toks) + 1))
	arr := (*[1 << 30]*C.char)(unsafe.Pointer(rawArr))[:len(toks)]
	defer C.free(unsafe.Pointer(rawArr))
	for i, t := range toks {
		arr[i] = C.CString(t)
	}

	tng := C.mitie_create_ner_training_instance(rawArr)
	if tng == nil {
		return nil, errors.New("unable to create training instance")
	}

	for _, v := range arr {
		C.free(unsafe.Pointer(v))
	}

	ti := &TrainingInstance{x: tng, tokens: toks}
	runtime.SetFinalizer(ti, freeTrainingInstance)
	return ti, nil
}

func freeTrainingInstance(t *TrainingInstance) {
	C.mitie_free(unsafe.Pointer(t.x))
}

// AddEntity adds an identified entity from the training instance.
func (t *TrainingInstance) AddEntity(start, length int, label string) error {
	if start < 0 {
		return fmt.Errorf("start value %d < 0", start)
	}
	clabel := C.CString(label)
	defer C.free(unsafe.Pointer(clabel))
	if C.mitie_add_ner_training_entity(t.x, C.ulong(start), C.ulong(length), clabel) != 0 {
		return fmt.Errorf("error adding entity: %d %d %s", start, length, label)
	}
	return nil
}
