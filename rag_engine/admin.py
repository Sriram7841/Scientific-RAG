from django.contrib import admin
from .models import TrainingRun, ModelMetrics, IndexBuildLog
@admin.register(TrainingRun)
class TRAdmin(admin.ModelAdmin):
    list_display = ('model_type','status','num_classes','train_size','test_size','started_at')
@admin.register(ModelMetrics)
class MMAdmin(admin.ModelAdmin):
    list_display = ('model_name','accuracy','precision','recall','f1_score','train_time_sec')
@admin.register(IndexBuildLog)
class IBAdmin(admin.ModelAdmin):
    list_display = ('status','total_documents','total_vectors','graph_nodes','started_at')
