require_relative "../test_helper"

class PoolingLayersTest < Minitest::Test
  def test_max_pool2d
    m = Torch::NN::MaxPool2d.new(3) #, stride: 2)
    m = Torch::NN::MaxPool2d.new([3, 2]) #, stride: [2, 1])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end

  def test_avg_pool2d
    m = Torch::NN::AvgPool2d.new(3) #, stride: 2)
    m = Torch::NN::AvgPool2d.new([3, 2]) #, stride: [2, 1])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end
end