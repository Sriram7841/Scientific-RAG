from django.db import models


class TrainingRun(models.Model):
    STATUS = [('pending','Pending'),('running','Running'),('done','Done'),('error','Error')]
    status       = models.CharField(max_length=10, choices=STATUS, default='pending')
    model_type   = models.CharField(max_length=10, default='both')
    train_size   = models.IntegerField(default=0)
    test_size    = models.IntegerField(default=0)
    num_classes  = models.IntegerField(default=0)
    error_msg    = models.TextField(blank=True)
    started_at   = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f'TrainingRun({self.model_type}, {self.status})'


class ModelMetrics(models.Model):
    training_run          = models.ForeignKey(TrainingRun, on_delete=models.CASCADE,
                                               related_name='metrics')
    model_name            = models.CharField(max_length=30)   # 'svm' | 'naive_bayes'
    accuracy              = models.FloatField(default=0)
    precision             = models.FloatField(default=0)
    recall                = models.FloatField(default=0)
    f1_score              = models.FloatField(default=0)
    train_time_sec        = models.FloatField(default=0)
    confusion_matrix_json = models.TextField(blank=True)
    class_report_json     = models.TextField(blank=True)
    created_at            = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.model_name}  Acc={self.accuracy:.3f}'


class IndexBuildLog(models.Model):
    STATUS = [('running','Running'),('done','Done'),('error','Error')]
    status          = models.CharField(max_length=10, choices=STATUS, default='running')
    total_documents = models.IntegerField(default=0)
    total_vectors   = models.IntegerField(default=0)
    graph_nodes     = models.IntegerField(default=0)
    graph_edges     = models.IntegerField(default=0)
    error_msg       = models.TextField(blank=True)
    started_at      = models.DateTimeField(auto_now_add=True)
    completed_at    = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-started_at']

    def __str__(self):
        return f'IndexBuild({self.status}, {self.total_vectors} vectors)'
