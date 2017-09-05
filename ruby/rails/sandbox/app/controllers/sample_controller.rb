class SampleController < ApplicationController

  def index
    # NOP
  end

  def new
    # NOP
  end

  def create
    render plain: "PARAS: key1=#{params[:data][:key1]}, key2=#{params[:data][:key2]}"
  end

  def query
    ret = ""

    ret << "# All Platforms\n"
    Platform.all.each { |platform|
      ret << "    " << platform.name << "\n"
    }

    ret << "# All PlatformModelRels\n"
    PlatformModelRel.all.each { |platformmodelrel|
      pfid = platformmodelrel.platform_id
      modelid = platformmodelrel.model_id
      platform = Platform.find(pfid)
      model = Model.find(modelid)
      ret << "    " << "#{platform.name} - #{model.name}\n"
    }

    ret << "# All Models\n"
    Model.all.each { |model|
      ret << "    " << model.name << "\n"
    }

    ret << "# All ModelColorRels\n"
    ModelColorRel.all.each { |modelcolorrel|
      modelid = modelcolorrel.model_id
      colorid = modelcolorrel.color_id
      model = Model.find(modelid)
      color = Color.find(colorid)
      ret << "    " << "#{model.name} - #{color.name}\n"
    }

    ret << "# All Colors\n"
    Color.all.each { |color|
      ret << "    " << color.name << "\n"
    }

    ret << "\n\n"



    render plain: ret
  end

end
