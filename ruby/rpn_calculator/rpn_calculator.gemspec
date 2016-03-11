# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'rpn_calculator/version'

Gem::Specification.new do |spec|
  spec.name          = "rpn_calculator"
  spec.version       = RpnCalculator::VERSION
  spec.authors       = ["Aki.SHIMO."]
  spec.email         = ["fezrestia@gmail.com"]
  spec.description   = %q{RPN calculator description}
  spec.summary       = %q{RPN calculator summary}
  spec.homepage      = ""
  spec.license       = "MIT"

  spec.files         = `git ls-files`.split($/)
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.3"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "rspec"
  spec.add_development_dependency "ariete"
end
