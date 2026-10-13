// Package ortruntime decides who owns the process-global ONNX Runtime environment.
package ortruntime

import (
	"errors"

	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

// ErrNotInitialized is returned when the caller owns the environment but has
// not initialized it.
var ErrNotInitialized = errors.New("ONNX Runtime is not initialized: set ORTLibrary, or initialize the environment before calling New")

// Init initializes the environment through ortenv for a non-empty library
// selector; ortenv retains it until process exit. An empty selector means the
// caller owns the environment, which must already be initialized.
func Init(library string) error {
	if library != "" {
		return ortenv.Init(library)
	}
	if !ort.IsInitialized() {
		return ErrNotInitialized
	}
	return nil
}
