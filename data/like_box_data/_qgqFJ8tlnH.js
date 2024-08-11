/*!CK:2522036236!*//*1442065415,*/

if (self.CavalryLogger) { CavalryLogger.start_js(["WPW9W"]); }

__d('PluginPageActionLogger',['BanzaiLogger','DOM','Event','PagePluginActions','PagePluginActionTypes'],function a(b,c,d,e,f,g,h,i,j,k,l){if(c.__markCompiled)c.__markCompiled();var m={initializeClickLoggers:function(n,o,p,q,r,s,t,u,v){var w=function(x,y){try{var aa=i.find(p,'.'+x);j.listen(aa,'click',function(ba){h.log('PagePluginActionsLoggerConfig',{page_id:n,page_plugin_action:y,page_plugin_action_type:l.CLICK,referer_url:o});});}catch(z){}};w(q,k.PAGE_LIKE);w(r,k.PAGE_UNLIKE);w(s,k.PAGE_AVATAR);w(t,k.PAGE_PERMALINK);w(u,k.PAGE_SHARE);w(v,k.PAGE_CTA);}};f.exports=m;},null);