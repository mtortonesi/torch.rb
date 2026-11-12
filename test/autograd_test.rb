require_relative "test_helper"

class AutogradTest < Minitest::Test
  def test_example
    x = Torch.ones(2, 2, requires_grad: true)
    y = x + 2
    z = y * y * 3
    out = z.mean
    out.backward
    assert_tensor [[4.5, 4.5], [4.5, 4.5]], x.grad
  end

  def test_example_backward
    x = Torch.randn(3, requires_grad: true)

    y = x * 2
    while y.data.norm < 1000
      y = y * 2
    end

    v = Torch.tensor([0.1, 1.0, 0.0001], dtype: :float)
    y.backward(v)

    assert_equal [3], x.grad.size
  end

  def test_requires_grad
    a = Torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    assert !a.requires_grad
    a.requires_grad!(true)
    assert a.requires_grad
    _b = (a * a).sum
  end

  def test_no_grad
    x = Torch.tensor([1.0], requires_grad: true)
    y = nil
    Torch.no_grad do
      y = x * 2
      assert !Torch.grad_enabled?
    end
    assert !y.requires_grad
    assert Torch.grad_enabled?
  end

  def test_enable_grad
    assert Torch.grad_enabled?
    Torch.no_grad do
      assert !Torch.grad_enabled?
      Torch.enable_grad do
        assert Torch.grad_enabled?
      end
    end
  end

  def test_grad
    x = Torch.tensor([1, 2, 3])
    assert_nil x.grad
  end

  def test_set_grad
    x = Torch.tensor([1, 2, 3])
    stress_gc do
      x.grad = Torch.tensor([1, 1, 1])
    end
    assert_tensor [1, 1, 1], x.grad
  end

  def test_set_grad_nil
    x = Torch.tensor([1, 2, 3])
    x.grad = nil
    assert_nil x.grad
  end

  def test_set_grad_different_size
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(ArgumentError) do
      x.grad = Torch.tensor([1, 1])
    end
    assert_equal "assigned grad has data of a different size", error.message
  end

  def test_set_grad_different_device
    skip "Requires CUDA" unless Torch::CUDA.available?

    x = Torch.tensor([1, 2, 3], device: "cuda")
    error = assert_raises(ArgumentError) do
      x.grad = Torch.tensor([1, 1, 1])
    end
    assert_equal "assigned grad has data of a different type", error.message
  end

  def test_set_grad_different_type
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(ArgumentError) do
      x.grad = Torch.tensor([1, 1, 1], dtype: :int32)
    end
    assert_equal "assigned grad has data of a different type", error.message
  end

  def test_variable
    x = Torch.tensor([1, 2, 3])
    v = nil
    assert_output(nil, /deprecated/) do
      v = Torch::Autograd::Variable.new(x)
    end
    assert x.eql?(v)
  end

  def test_variable_invalid
    error = assert_raises(ArgumentError) do
      Torch::Autograd::Variable.new(Object.new)
    end
    assert_equal "Variable data has to be a tensor, but got Object", error.message
  end

  def test_tensor_requires_grad
    [[1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]]].each do |data|
      x = Torch.tensor(data, requires_grad: true)
      assert x.requires_grad
      assert x.leaf?
      (x * x).sum.backward
      assert_equal Torch.tensor(data).mul(2).to_a, x.grad.to_a
    end
  end

  def test_tensor_requires_grad_scalar
    x = Torch.tensor(3.0, requires_grad: true)
    assert x.leaf?
    (x * x).backward
    assert_in_delta 6.0, x.grad.item
  end

  def test_tensor_requires_grad_dtype
    x = Torch.tensor([1.5, 2.5], dtype: :float64, requires_grad: true)
    assert_equal :float64, x.dtype
    assert x.leaf?
    x.sum.backward
    assert_equal [1.0, 1.0], x.grad.to_a
    assert_equal :float64, x.grad.dtype
  end

  def test_tensor_requires_grad_device
    devices = ["cpu"]
    devices << "mps" if Torch::Backends::MPS.available?
    devices << "cuda" if Torch::CUDA.available?
    devices.each do |device|
      x = Torch.tensor([1.0, 2.0], device: device, requires_grad: true)
      assert x.leaf?, "not a leaf on #{device}"
      (x * x).sum.backward
      assert_equal [2.0, 4.0], x.grad.cpu.to_a
    end
  end

  def test_tensor_requires_grad_false
    [Torch.tensor([1.0, 2.0]), Torch.tensor([1.0, 2.0], requires_grad: false)].each do |x|
      assert !x.requires_grad
      assert x.leaf?
    end
  end

  # 1.7.0 behavior
  def test_max
    a = Torch.tensor([3.0, 2, 3], requires_grad: true)
    a.max.backward
    assert_equal [0.5, 0, 0.5], a.grad.to_a
  end
end
