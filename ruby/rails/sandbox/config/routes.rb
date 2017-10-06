Rails.application.routes.draw do
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html

  root 'sample#index'

  get   '/sample/new',                    to: 'sample#new',                           as: 'new_sample'
  post  '/sample',                        to: 'sample#create',                        as: 'create_sample'

  get   '/query',                         to: 'sample#query',                         as: 'query'
  get   '/test_param',                    to: 'sample#test_param',                    as: 'test_param'

  get   '/d3sample',                      to: 'sample#d3sample',                      as: 'd3sample'
  get   '/d3sample_data',                 to: 'sample#d3sample_data',                 as: 'd3sample_data'
  get   '/d3_data_dependency_wheel',      to: 'sample#d3_data_dependency_wheel',      as: 'd3_data_dependency_wheel'

end
