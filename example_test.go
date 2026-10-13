package phonmatchnet_test

import (
	"context"
	"fmt"
	"log"

	phonmatchnet "github.com/androiddrew/go-phonmatchnet"
	ort "github.com/yalue/onnxruntime_go"
)

// loadClip stands in for the application's audio decoding. The engine requires
// one 0.5-2 second clip of 16 kHz mono float32 PCM in [-1, 1].
func loadClip() []float32 { return make([]float32, phonmatchnet.SampleRate) }

// Detect a keyword in one clip. The models are embedded, so no paths are needed.
func Example() {
	engine, err := phonmatchnet.New(phonmatchnet.Config{
		ORTLibrary: "libonnxruntime.so", // OS loader name or a path to the library
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	ctx := context.Background()
	keyword, err := engine.Keyword(ctx, "hey computer") // reuse across clips
	if err != nil {
		log.Fatal(err)
	}
	result, err := engine.Detect(ctx, phonmatchnet.Request{
		Samples: loadClip(), SampleRate: phonmatchnet.SampleRate, Keyword: keyword,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("score %.3f accepted %v\n", result.Score, result.Accepted)
}

// Choose the more sensitive calibrated operating point, or compare
// Result.Score with a threshold validated on your own recordings.
func ExampleEngine_Thresholds() {
	engine, err := phonmatchnet.New(phonmatchnet.Config{ORTLibrary: "libonnxruntime.so"})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	for far, threshold := range engine.Thresholds() {
		fmt.Printf("target FAR %g: score >= %.4f\n", far, threshold)
	}
	keyword, err := engine.Keyword(context.Background(), "okay naptrap")
	if err != nil {
		log.Fatal(err)
	}
	result, err := engine.Detect(context.Background(), phonmatchnet.Request{
		Samples: loadClip(), SampleRate: phonmatchnet.SampleRate, Keyword: keyword, FAR: 0.1,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.Accepted)
}

// An application that already initializes ONNX Runtime itself leaves
// ORTLibrary empty. The engine then uses that environment and never destroys it.
func Example_callerOwnedEnvironment() {
	ort.SetSharedLibraryPath("libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment() // runs after engine.Close

	engine, err := phonmatchnet.New(phonmatchnet.Config{})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	fmt.Println(engine.Initialization().RuntimeVersion)
}
