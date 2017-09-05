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

    log = "## All Platforms"
    ret << log << "\n"
    logger.debug(log)
    Platform.all.each { |platform|
      ret << "    " << platform.name << "\n"
    }

    ret << "\n"

    log = "## All PlatformModelRels"
    ret << log << "\n"
    logger.debug(log)
    PlatformModelRel.all.each { |platformmodelrel|
      pfid = platformmodelrel.platform_id
      modelid = platformmodelrel.model_id
      platform = Platform.find(pfid)
      model = Model.find(modelid)
      ret << "    " << "#{platform.name} - #{model.name}\n"
    }

    ret << "\n"

    log = "## All Models"
    ret << log << "\n"
    logger.debug(log)
    Model.all.each { |model|
      ret << "    " << model.name << "\n"
    }

    ret << "\n"

    log = "## All ModelColorRels"
    ret << log << "\n"
    logger.debug(log)
    ModelColorRel.all.each { |modelcolorrel|
      modelid = modelcolorrel.model_id
      colorid = modelcolorrel.color_id
      model = Model.find(modelid)
      color = Color.find(colorid)
      ret << "    " << "#{model.name} - #{color.name}\n"
    }

    ret << "\n"

    log = "## All Colors"
    ret << log << "\n"
    logger.debug(log)
    Color.all.each { |color|
      ret << "    " << color.name << "\n"
    }

    ret << "\n"

    log = "## Platform.joins(:models)"
    ret << log << "\n"
    logger.debug(log)
    Platform.joins(:models).each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Platform.joins(:models).select(\"platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Platform.joins(:models)
        .select("platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Platform.includes(:models).select(\"platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Platform.includes(:models)
        .references(:models)
        .select("platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name")
        .where('models.name = "Rachael"')
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Platform.joins({ :models => :colors}).select(\"platforms.*, models.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Platform.joins({ :models => :colors})
        .select("platforms.*, models.*, colors.*, platforms.name AS platform_name, models.name AS model_name, colors.name AS color_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"

    log = "## Color.joins({ :models => :platform}).select(\"platforms.*, models.*, coloros.*, platforms.name AS platform_name, models.name AS model_name\")"
    ret << log << "\n"
    logger.debug(log)
    Color.joins({ :models => :platform})
        .select("colors.*, models.*, platforms.*, platforms.name AS platform_name, models.name AS model_name, colors.name AS color_name")
        .each { |record|
      attrs = record.attributes
      attrs.delete('created_at')
      attrs.delete('updated_at')
      ret << "    " << attrs.to_s << "\n"
    }

    ret << "\n"






    render plain: ret
  end

end
