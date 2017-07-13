### Bazel Build commands

```
1. Train and Save the Model:
  > bazel build //tensorflow_serving/example:sentiment_saved_model
  > bazel-bin/tensorflow_serving/example/sentiment_saved_model /tmp/sentiment_model

2. Start Server
  > bazel build //tensorflow_serving/model_servers:tensorflow_model_server
  > bazel-bin/tensorflow_serving/example/sentiment_client --num_tests=1000 --server=localhost:9000

3. Start Client
  > bazel build //tensorflow_serving/example:sentiment_client
  > bazel-bin/tensorflow_serving/example/sentiment_client --num_tests=1000 --server=localhost:9000

```
### Reference

* [Common Bazel Build Installation Error](https://github.com/tensorflow/serving/issues/421)
