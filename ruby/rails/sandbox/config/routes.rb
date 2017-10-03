Rails.application.routes.draw do
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html

  root 'sample#index'

  get   '/sample/new',    to: 'sample#new',           as: 'new_sample'
  post  '/sample',        to: 'sample#create',        as: 'create_sample'

  get   '/query',         to: 'sample#query',         as: 'query'
  get   '/test_param',    to: 'sample#test_param',    as: 'test_param'


end
